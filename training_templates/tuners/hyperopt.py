from abc import ABC, abstractmethod
from typing import Dict, Callable, Any

from hyperopt import STATUS_OK
from hyperopt import Trials, fmin, tpe, space_eval
from hyperopt.early_stop import no_progress_loss

import numpy as np
import pandas as pd

from training_templates.metrics import classification_metrics
from training_templates.mlflow import mlflow_disable_autolog
from training_templates.constants import BOOSTED_MODELS


class Tuner(ABC):
    def __init__(
        self,
        *,
        hyperparameter_space: Dict[str, Any],
        hyperopt_max_evals: int,
        hyperopt_iteration_stop_count: int,
        hyperopt_early_stopping_threshold: float,
    ):
        self.hyperparameter_space = hyperparameter_space
        self.hyperopt_max_evals = hyperopt_max_evals
        self.hyperopt_iteration_stop_count = hyperopt_iteration_stop_count
        self.hyperopt_early_stopping_threshold = hyperopt_early_stopping_threshold

    @abstractmethod
    def config_hyperopt_objective_fn(self) -> Callable:
        """
        Define the hyperopt objective function to minimize
        """

    @mlflow_disable_autolog
    def tune(
        self,
        init_model: Callable,
        X_train_transformed: pd.DataFrame,
        X_val_transformed: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        random_state: int,
    ):
        self.init_model = init_model
        self.X_train_transformed = X_train_transformed
        self.X_val_transformed = X_val_transformed
        self.y_train = y_train
        self.y_val = y_val

        object_fn = self.config_hyperopt_objective_fn()

        trials = Trials()

        print("Beginning hyperopt parameter search")
        early_stop_fn = no_progress_loss(
            iteration_stop_count=self.hyperopt_iteration_stop_count,
            percent_increase=self.hyperopt_early_stopping_threshold,
        )

        best_params = fmin(
            fn=object_fn,
            space=self.hyperparameter_space,
            algo=tpe.suggest,
            max_evals=self.hyperopt_max_evals,
            trials=trials,
            rstate=np.random.default_rng(random_state),
            early_stop_fn=early_stop_fn,
        )

        best_params = space_eval(self.hyperparameter_space, best_params)

        if type(init_model()) in BOOSTED_MODELS:
            best_params["n_estimators"] = trials.best_trial["result"]["metrics"][
                "best_iteration"
            ]

            to_remove = ["eval_metric", "early_stopping_rounds"]
            for param in to_remove:
                best_params.pop(param, None)

        print("Best model parameters:")
        for param, value in best_params.items():
            print(param, value)

        print("\nBest model statistics:")
        for metric, value in trials.best_trial["result"]["metrics"].items():
            print(metric, value)

        return best_params


class XGBoostHyperoptTuner(Tuner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def config_hyperopt_objective_fn(self) -> Callable:
        """
        Return an Hyperopt objective function for an XGBoost model that leveraging early
        stopping
        """

        def hyperopt_objective_fn(params):
            model = self.init_model(params)

            model.fit(
                self.X_train_transformed,
                self.y_train.values.ravel(),
                eval_set=[(self.X_val_transformed, self.y_val.values.ravel())],
                verbose=False,
            )

            best_score = model.best_score
            metrics = classification_metrics(model, self.X_val_transformed, self.y_val)

            return {"status": STATUS_OK, "loss": 1 - best_score, "metrics": metrics}

        return hyperopt_objective_fn


class SkLearnHyperoptTuner(Tuner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def config_hyperopt_objective_fn(self) -> Callable:
        """
        A standard tuner compatible with a wide variety of scikit-learn
        """

        def hyperopt_objective_fn(params):
            model = self.init_model(params)

            model.fit(
                self.X_train_transformed,
                self.y_train.values.ravel(),
            )

            metrics = classification_metrics(model, self.X_val_transformed, self.y_val)

            return {
                "status": STATUS_OK,
                "loss": 1 - metrics["f1_val"],
                "metrics": metrics,
            }

        return hyperopt_objective_fn
