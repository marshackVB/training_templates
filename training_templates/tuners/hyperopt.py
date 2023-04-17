from abc import ABC, abstractmethod
from typing import Dict, Union, Callable, Any

from hyperopt import STATUS_OK
from hyperopt import Trials, fmin, tpe, space_eval
from hyperopt.early_stop import no_progress_loss
import numpy as np
import pandas as pd

from training_templates.metrics import xgboost_classification_metrics, sklearn_classification_metrics
from training_templates.tuners.formatters import format_hyperopt_sklearn, format_hyperopt_xbgoost
from training_templates.mlflow import mlflow_disable_autolog


class Tuner(ABC):
    @abstractmethod
    def tune(self) -> Dict[str, Union[str, int, float]]:
        """
        Execute hyperparameter tuning strategy; return a dictionary containing the
        best model parameters.
        """


class HyperoptTuner(Tuner, ABC):
    def __init__(self, *, hyperparameter_space: Dict[str, Any], hyperopt_max_evals: int, hyperopt_iteration_stop_count: int, 
                 hyperopt_early_stopping_threshold: float):
        
        self.hyperparameter_space = hyperparameter_space
        self.hyperopt_max_evals = hyperopt_max_evals
        self.hyperopt_iteration_stop_count = hyperopt_iteration_stop_count
        self.hyperopt_early_stopping_threshold = hyperopt_early_stopping_threshold

    @abstractmethod
    def config_hyperopt_objective_fn(self, init_model: Callable, X_train_transformed: pd.DataFrame, 
                                     X_val_transformed: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> Callable:
        """
        Return a hyperopt object function
        """
        

class XGBoostHyperoptTuner(HyperoptTuner):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def config_hyperopt_objective_fn(self, init_model, X_train_transformed, X_val_transformed,
                                     y_train, y_val):
        """
        Return an Hyperopt objective function for an XGBoost model that leveraging early
        stopping
        """

        def hyperopt_objective_fn(params):
            params = format_hyperopt_xbgoost(params)

            model = init_model(params)

            model.fit(
                X_train_transformed,
                y_train.values.ravel(),
                eval_set=[(X_val_transformed, y_val.values.ravel())],
                verbose=False,
            )

            best_score = model.best_score
            metrics = xgboost_classification_metrics(model, X_val_transformed, y_val)

            return {"status": STATUS_OK, "loss": 1 - best_score, "metrics": metrics}

        return hyperopt_objective_fn
    

    @mlflow_disable_autolog
    def tune(self, init_model, X_train_transformed, X_val_transformed, 
             y_train, y_val, random_state):
    
        object_fn = self.config_hyperopt_objective_fn(init_model, X_train_transformed, X_val_transformed, 
                                                      y_train, y_val)

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
        best_params["n_estimators"] = trials.best_trial["result"]["metrics"][
            "best_iteration"
        ]
        final_model_parameters = format_hyperopt_xbgoost(best_params)

        to_remove = ["eval_metric", "early_stopping_rounds"]
        for param in to_remove:
            final_model_parameters.pop(param, None)

        print("Best model parameters:")
        for param, value in final_model_parameters.items():
            print(param, value)

        print("\nBest model statistics:")
        for metric, value in trials.best_trial["result"]["metrics"].items():
            print(metric, value)

        return final_model_parameters
    

class SkLearnHyperoptTuner(HyperoptTuner):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def config_hyperopt_objective_fn(self, init_model, X_train_transformed, X_val_transformed, 
                                     y_train, y_val) ->Callable:
        """
        A standard tuner compatible with a wide variety of scikit-learn models
        """

        def hyperopt_objective_fn(params):

            params = format_hyperopt_sklearn(params, init_model())

            model = init_model(params)

            model.fit(
                X_train_transformed,
                y_train.values.ravel(),
            )

            metrics = sklearn_classification_metrics(model, X_val_transformed, y_val)

            return {"status": STATUS_OK, "loss": 1 - metrics["f1_val"], "metrics": metrics}

        return hyperopt_objective_fn
    

    @mlflow_disable_autolog
    def tune(self, init_model, X_train_transformed, X_val_transformed, 
             y_train, y_val, random_state):
 
        object_fn = self.config_hyperopt_objective_fn(init_model, X_train_transformed, X_val_transformed, y_train, y_val)

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
        final_model_parameters = format_hyperopt_xbgoost(best_params)

        print("Best model parameters:")
        for param, value in final_model_parameters.items():
            print(param, value)

        print("\nBest model statistics:")
        for metric, value in trials.best_trial["result"]["metrics"].items():
            print(metric, value)

        return final_model_parameters



