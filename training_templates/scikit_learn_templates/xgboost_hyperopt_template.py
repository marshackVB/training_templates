from collections import OrderedDict

from hyperopt import STATUS_OK
from hyperopt import Trials, fmin, tpe, space_eval
from hyperopt.early_stop import no_progress_loss
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb

from training_templates.scikit_learn_templates.base_templates import SkLearnHyperoptBase


class XGBoostHyperoptTrainer(SkLearnHyperoptBase):
    """
    Implements a Hyperopt objective function for an XGBoost training workflow.
    """

    convert_to_int = [
        "n_estimators",
        "max_depth",
        "max_leaves",
        "max_bin",
        "grow_policy",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(model=xgb.XGBClassifier, model_name="xgboost", *args, **kwargs)

    def format_hyperopt_for_sklearn(self, hyperopt_params):
        for param, value in hyperopt_params.items():
            if param in self.__class__.convert_to_int:
                hyperopt_params[param] = int(value)

        return hyperopt_params

    def config_hyperopt_objective_fn(self, X_train_transformed, X_val_transformed):
        """
        Return an Hyperopt objective function for an XGBoost model that leveraging early
        stopping
        """

        def hyperopt_objective_fn(params):
            params = self.format_hyperopt_for_sklearn(params)

            model = self.init_model(params)

            model.fit(
                X_train_transformed,
                self.y_train.values.ravel(),
                eval_set=[(X_val_transformed, self.y_val.values.ravel())],
                verbose=False,
            )

            best_score = model.best_score
            best_iteration = model.best_iteration
            best_xgboost_rounds = (0, best_iteration + 1)

            precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
                self.y_val,
                model.predict(X_val_transformed, iteration_range=best_xgboost_rounds),
                average="weighted",
            )

            digits = 3
            metrics = OrderedDict()
            metrics["precision_val"] = round(precision_val, digits)
            metrics["recall_val"] = round(recall_val, digits)
            metrics["f1_val"] = round(f1_val, digits)
            metrics["best_iteration"] = best_iteration

            return {"status": STATUS_OK, "loss": 1 - best_score, "metrics": metrics}

        return hyperopt_objective_fn

    def tune_hyperparameters(self):
        """
        Launch the Hyperopt Trials workflow
        """
        X_train_transformed, X_val_transformed = self.transform_features_for_hyperopt()
        object_fn = self.config_hyperopt_objective_fn(
            X_train_transformed, X_val_transformed
        )

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
            rstate=np.random.default_rng(self.random_state),
            early_stop_fn=early_stop_fn,
        )

        best_params = space_eval(self.hyperparameter_space, best_params)
        best_params["n_estimators"] = trials.best_trial["result"]["metrics"][
            "best_iteration"
        ]
        final_model_parameters = self.format_hyperopt_for_sklearn(best_params)

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
