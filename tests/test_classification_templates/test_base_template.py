import copy
from collections import OrderedDict

import mlflow
import numpy as np
import pandas as pd
import pytest
from hyperopt import STATUS_OK, hp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from training_templates import RandomForestHyperoptTrainer


hyperparameter_space = {"n_estimators": hp.quniform("n_estimators", 2, 10, 1),
                        "max_features": hp.uniform("max_features", 0.5, 1.0),
                        "criterion": hp.choice("criterion", ["gini", "entropy"])}


def test_join_to_pandas(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    assert isinstance(trainer.feature_df, pd.DataFrame)
    assert trainer.feature_df.shape[0] > 0


def test_types_split_train_val_test(spark, feature_table, training_args):
    """
    Feature tables are Pandas DataFrames and label tables are
    Pandas Series
    """

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    trainer.train_test_split()

    assert isinstance(trainer.X_train, pd.core.frame.DataFrame)
    assert isinstance(trainer.X_val, pd.core.frame.DataFrame)

    assert isinstance(trainer.y_train, pd.core.frame.Series)
    assert isinstance(trainer.y_val, pd.core.frame.Series)


@pytest.mark.parametrize(
    "spark, feature_table, training_args, train_size, random_state, shuffle",
    [
        ("spark", "feature_table", "training_args", 0.7, 123, False),
        ("spark", "feature_table", "training_args", 0.6, 123, True),
        ("spark", "feature_table", "training_args", 0.5, 999, True),
    ], indirect=["spark", "feature_table", "training_args"]
)
def test_proportions_split_train_val_test(
    spark, feature_table, training_args, train_size, random_state, shuffle, request
):

    training_args["hyperparameter_space"] = hyperparameter_space
    training_args["train_size"] = train_size
    training_args["random_state"] = random_state
    training_args["train_eval_shuffle"] = shuffle

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
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


def test_no_obs_overlap_split_train_val_test(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    trainer.train_test_split()

    common_train_observations = set(trainer.X_train.index) & set(trainer.X_val.index)
    common_val_observations = set(trainer.y_train.index) & set(trainer.y_val.index)

    assert len(common_train_observations) == 0
    assert len(common_val_observations) == 0


def test_fit_predict_model_pipeline(spark, feature_table, training_args):
    """
    Model training pipeline component types are correct; all column types
    are passed to the model pipeline; model pipeline prediction works
    as expected.
    """
    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
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


def test_format_hyperopt_for_sklearn(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    class DummyModel():
        def __init__(self):
            self.a = 1
            self.b = 1.0
            self.c = "c"
            self.d = 10

    trainer.model = DummyModel

    hyperopt_params = {"a": 9.0,
                       "b": 10.0,
                       "c": "c",
                       "d": 100.0}

    formatted_params = trainer.format_hyperopt_for_sklearn(hyperopt_params)

    expected_params = {"a": 9, 
                       "b": 10.0, 
                       "c": 'c', 
                       "d": 100}

    assert formatted_params == expected_params


def test_hyperopt_objective_fn(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
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


def test_hyperopt_search(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    trainer.train_test_split()

    best_parameters = trainer.tune_hyperparameters()

    assert isinstance(best_parameters, dict)


def test_train(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = RandomForestHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )
    
    trainer.train()

    logged_model = f"runs:/{trainer.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    predictions = loaded_model.predict(trainer.X_val)

    observations = trainer.X_val.shape[0]
    assert predictions.shape[0] == (observations)
