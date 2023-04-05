from collections import OrderedDict
import copy

from hyperopt import hp
import mlflow
import numpy as np

from training_templates import XGBoostHyperoptTrainer


hyperparameter_space = {"max_depth": hp.quniform("max_depth", 2, 10, 1),
                        'eval_metric': 'auc',
                        'early_stopping_rounds': 50}


def test_format_hyperopt_for_sklearn(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = XGBoostHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    hyperopt_params = {"a": 1.0,
                       "b": "b",
                       "c": 1,
                       "d": 10.0,
                       "e": 10.0}

    convert_to_int = ["a", "d", "e"]

    trainer.convert_to_int = convert_to_int

    formated_params = trainer.format_hyperopt_for_sklearn(hyperopt_params)

    expected_result = {"a": 1,
                       "b": "b",
                       "c": 1,
                       "d": 10,
                       "e": 10.0}
    
    assert formated_params == expected_result


def test_hyperopt_objective_fn(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = XGBoostHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    trainer.train_test_split()

    X_train_transformed, X_val_transformed = trainer.transform_features_for_hyperopt()

    objective_fn = trainer.config_hyperopt_objective_fn(
        X_train_transformed, X_val_transformed
    )

    params = {"max_depth": 2,
              "eval_metric": 'auc',
              "early_stopping_rounds": 50}
    
    objective_results = objective_fn(params)

    loss_type = type(objective_results["loss"])
    assert loss_type == float or loss_type == np.float64
    assert objective_results["status"] == "ok"
    assert isinstance(objective_results["metrics"], OrderedDict)


def test_hyperopt_search(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = XGBoostHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )

    trainer.train_test_split()

    best_parameters = trainer.tune_hyperparameters()

    assert isinstance(best_parameters, dict)


def test_train(spark, feature_table, training_args):

    training_args["hyperparameter_space"] = hyperparameter_space

    trainer = XGBoostHyperoptTrainer(
        feature_table, f"{feature_table}_train", **training_args
    )
    trainer.train()

    logged_model = f"runs:/{trainer.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    predictions = loaded_model.predict(trainer.X_val)

    observations = trainer.X_val.shape[0]
    assert predictions.shape[0] == (observations)