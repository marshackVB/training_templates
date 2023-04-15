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


@pytest.fixture
def trainer(spark, feature_table, training_args):
    hyperparameter_space = {
        "n_estimators": hp.quniform("n_estimators", 2, 10, 1),
        "max_features": hp.uniform("max_features", 0.5, 1.0),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }

    training_args["hyperparameter_space"] = hyperparameter_space
    training_args["delta_feature_table"] = feature_table
    training_args["delta_train_val_id_table"] = f"{feature_table}_train"

    trainer = RandomForestHyperoptTrainer(
        **training_args
    )
    return trainer


def test_join_to_pandas(trainer):
    """
    Spark DataFrames of feature values and test/val record ids are
    joined and converted to a Pandas DataFrame.
    """
    assert isinstance(trainer.feature_df, pd.DataFrame)
    assert trainer.feature_df.shape[0] > 0


def test_types_split_train_val_test(trainer):
    """
    Pandas DataFrame of features values are properly split into training
    and validation datasets
    """
    trainer.train_test_split()

    assert isinstance(trainer.X_train, pd.core.frame.DataFrame)
    assert isinstance(trainer.X_val, pd.core.frame.DataFrame)

    assert isinstance(trainer.y_train, pd.core.frame.Series)
    assert isinstance(trainer.y_val, pd.core.frame.Series)


@pytest.mark.parametrize(
    "trainer, train_size, random_state, shuffle",
    [
        ("trainer", 0.7, 123, False),
        ("trainer", 0.6, 123, True),
        ("trainer", 0.5, 999, True),
    ],
    indirect=["trainer"],
)
def test_proportions_split_train_val_test(trainer, train_size, random_state, shuffle):
    """
    Training and validation dataset split proportions are as expected
    """
    trainer.train_size = train_size
    trainer.random_state = random_state
    trainer.train_eval_shuffle = shuffle

    trainer.train_test_split()

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


def test_no_obs_overlap_split_train_val_test(trainer):
    """
    Training and validation dataset records do not overlap.
    """
    trainer.train_test_split()

    common_train_observations = set(trainer.X_train.index) & set(trainer.X_val.index)
    common_val_observations = set(trainer.y_train.index) & set(trainer.y_val.index)

    assert len(common_train_observations) == 0
    assert len(common_val_observations) == 0


def test_fit_predict_model_pipeline(trainer):
    """
    Model training pipeline component types are correct; all column types
    are passed to the model pipeline; model pipeline prediction works
    as expected.
    """
    trainer.train_test_split()
    model_params = {"n_estimators": 10}
    model_pipeline = trainer.init_model_pipeline(model_params)

    assert isinstance(model_pipeline["preprocessing_pipeline"], ColumnTransformer)
    assert isinstance(model_pipeline["model"], RandomForestClassifier)
    assert isinstance(model_pipeline, Pipeline)

    model_pipeline.fit(trainer.X_train, trainer.y_train)
    predictions = model_pipeline.predict_proba(trainer.X_val)
    observations = trainer.X_val.shape[0]
    assert predictions.shape == (observations, 2)


def test_format_hyperopt_for_sklearn(trainer):
    """
    Hyperopt parameter types are properly converted to scikit-learn's
    expected types when they differ.
    """

    class DummyModel:
        def __init__(self):
            self.a = 1
            self.b = 1.0
            self.c = "c"
            self.d = 10

    trainer.model = DummyModel

    hyperopt_params = {"a": 9.0, "b": 10.0, "c": "c", "d": 100.0}

    formatted_params = trainer.format_hyperopt_for_sklearn(hyperopt_params)

    expected_params = {"a": 9, "b": 10.0, "c": "c", "d": 100}

    assert formatted_params == expected_params


def test_hyperopt_objective_fn(trainer):
    """
    The hyperopt objective function returns a dictionary with the
    expected data types.
    """
    trainer.train_test_split()

    X_train_transformed, X_val_transformed = trainer.transform_features_for_hyperopt()
    objective_fn = trainer.config_hyperopt_objective_fn(
        X_train_transformed, X_val_transformed
    )

    model_params = {"n_estimators": 10}
    objective_results = objective_fn(model_params)

    assert isinstance(objective_results, dict)
    assert objective_results["status"] == "ok"
    assert type(objective_results["loss"]) == np.float64
    assert isinstance(objective_results["metrics"], OrderedDict)


def test_hyperopt_search(trainer):
    """
    Hyperopt tuning over multiple iterations executes and returns
    a dictionary.
    """
    trainer.train_test_split()
    best_parameters = trainer.tune_hyperparameters()
    assert isinstance(best_parameters, dict)


def test_train(trainer):
    """
    The full training workflow executes and logs a model to MLflow that
    can be loaded an scores data.
    """
    trainer.train()
    logged_model = f"runs:/{trainer.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    predictions = loaded_model.predict(trainer.X_val)
    observations = trainer.X_val.shape[0]
    assert predictions.shape[0] == (observations)
