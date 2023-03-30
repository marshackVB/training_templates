import copy
from collections import OrderedDict

import mlflow
import numpy as np
import pandas as pd
import pytest
from hyperopt import STATUS_OK, fmin, hp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from training_templates import SkLearnHyperoptBase


class TestSklearnHyperoptBase(SkLearnHyperoptBase):
    """
    A subclass designed to test the functionality of SkLearnHyperoptBase
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(RandomForestClassifier,
                         "random_forecast",
                         *args, 
                         **kwargs)

    def config_hyperopt_objective_fn(self, X_train_transformed, X_val_transformed):
        def hyperopt_objective_fn(params):
            params["n_estimators"] = int(params["n_estimators"])

            model = self.init_model(params)

            model.fit(X_train_transformed, self.y_train.values.ravel())

            precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
                self.y_val, model.predict(X_val_transformed), average="weighted"
            )

            digits = 3
            metrics = OrderedDict()
            metrics["precision_val"] = round(precision_val, digits)
            metrics["recall_val"] = round(recall_val, digits)
            metrics["f1_val"] = round(f1_val, digits)

            return {"status": STATUS_OK, "loss": 1 - f1_val, "metrics": metrics}

        return hyperopt_objective_fn


training_args = {
    "train_size": 0.8,
    "numerical_cols": ["Age", "FareRounded"],
    "categorical_cols": [
        "NamePrefix",
        "Sex",
        "CabinChar",
        "CabinMulti",
        "Embarked",
        "Parch",
        "Pclass",
        "SibSp",
    ],
    "binary_cols": ["NameMultiple"],
    "label_col": "Survived",
    "problem_type": "classification",
    "hyperparameter_space": {"n_estimators": hp.quniform("n_estimators", 2, 10, 1)},
    "hyperopt_max_evals": 5,
    "hyperopt_iteration_stop_count": 1,
    "hyperopt_early_stopping_threshold": 0.05,
    "mlflow_experiment_location": "/Shared/ml_production_experiment",
    "mlflow_run_description": "A Model instance designed for testing",
    "random_state": 123,
    "train_eval_shuffle": True,
    "commit_hash": None,
    "release_version": None,
}


def test_join_to_pandas(spark, feature_table):
    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )

    assert isinstance(trainer.feature_df, pd.DataFrame)
    assert trainer.feature_df.shape[0] > 0


def test_types_split_train_val_test(spark, feature_table):
    """
    Feature tables are Pandas DataFrames and label tables are
    Pandas Series
    """

    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train_test_split()

    assert isinstance(trainer.X_train, pd.core.frame.DataFrame)
    assert isinstance(trainer.X_val, pd.core.frame.DataFrame)

    assert isinstance(trainer.y_train, pd.core.frame.Series)
    assert isinstance(trainer.y_val, pd.core.frame.Series)


@pytest.mark.parametrize(
    "spark, feature_table, train_size, random_state, shuffle",
    [
        ("spark", "feature_table", 0.7, 123, False),
        ("spark", "feature_table", 0.6, 123, True),
        ("spark", "feature_table", 0.5, 999, True),
    ],
)
def test_proportions_split_train_val_test(
    spark, feature_table, train_size, random_state, shuffle, request
):
    spark = request.getfixturevalue(spark)
    feature_table = request.getfixturevalue(feature_table)

    updated_args = copy.deepcopy(training_args)
    updated_args["train_size"] = train_size
    updated_args["random_state"] = random_state
    updated_args["train_eval_shuffle"] = shuffle

    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **updated_args
    )

    trainer.train_test_split()

    # Train / val ratio is as expected
    X_train_cnt = trainer.X_train.shape[0]
    X_val_cnt = trainer.X_val.shape[0]
    y_train_cnt = trainer.y_train.shape[0]
    y_val_cnt = trainer.y_val.shape[0]
    combined_cnt = trainer.feature_df.shape[0]

    val_size = round(1 - train_size, 1)
    assert round(X_train_cnt / combined_cnt, 1) == train_size
    assert round(X_val_cnt / combined_cnt, 1) == val_size
    assert round(y_train_cnt / combined_cnt, 1) == train_size
    assert round(y_val_cnt / combined_cnt, 1) == val_size


def test_no_obs_overlap_split_train_val_test(spark, feature_table):
    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train_test_split()

    common_train_observations = set(trainer.X_train.index) & set(trainer.X_val.index)
    common_val_observations = set(trainer.y_train.index) & set(trainer.y_val.index)

    assert len(common_train_observations) == 0
    assert len(common_val_observations) == 0


def test_fit_predict_model_pipeline(spark, feature_table):
    """
    Model training pipeline component types are correct; all column types
    are passed to the model pipeline; model pipeline prediction works
    as expected.
    """
    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train_test_split()

    model_params = {"n_estimators": 10}
    model_pipeline = trainer.init_model_pipeline(model_params)

    # Pipeline and component types are as expected
    assert isinstance(model_pipeline["preprocessing_pipeline"], ColumnTransformer)
    assert isinstance(model_pipeline["model"], RandomForestClassifier)
    assert isinstance(model_pipeline, Pipeline)

    # Numerical, categorical, and binary columns are all passed to ColumnTransformer
    column_transformations = {}
    for transformer in model_pipeline["preprocessing_pipeline"].transformers:
        column_transformations[transformer[0]] = transformer[2]

    for column_type, list_of_columns in column_transformations.items():
        assert list_of_columns == training_args[column_type]

    model_pipeline.fit(trainer.X_train, trainer.y_train)
    predictions = model_pipeline.predict_proba(trainer.X_val)
    observations = trainer.X_val.shape[0]
    assert predictions.shape == (observations, 2)


def test_hyperopt_objective_fn(spark, feature_table):
    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train_test_split()

    X_train_transformed, X_val_transformed = trainer.transform_features_for_hyperopt()
    objective_fn = trainer.config_hyperopt_objective_fn(
        X_train_transformed, X_val_transformed
    )

    model_params = {"n_estimators": 10}
    objective_results = objective_fn(model_params)

    assert objective_results["status"] == "ok"
    assert type(objective_results["loss"]) == np.float64
    assert isinstance(objective_results["metrics"], OrderedDict)


def test_hyperopt_search(spark, feature_table):
    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train_test_split()

    best_parameters = trainer.tune_hyperparameters()

    assert isinstance(best_parameters, dict)


def test_train(spark, feature_table):
    trainer = TestSklearnHyperoptBase(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train()

    logged_model = f"runs:/{trainer.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    predictions = loaded_model.predict(trainer.X_val)

    observations = trainer.X_val.shape[0]
    assert predictions.shape == (observations,)
