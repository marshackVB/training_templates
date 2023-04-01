from collections import OrderedDict

import xgboost as xgb
from hyperopt import STATUS_OK
from sklearn.metrics import precision_recall_fscore_support

from training_templates.scikit_learn_templates.base_templates import SkLearnHyperoptBase


class XGBoostHyperoptTrainer(SkLearnHyperoptBase):
    """
    Implements a Hyperopt objective function for an XGBoost training workflow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(xgb.XGBClassifier, "xgboost", *args, **kwargs)

    def config_hyperopt_objective_fn(self, X_train_transformed, X_val_transformed):
        """
        Return an Hyperopt objective function for an XGBoost model that leveraging early
        stopping
        """

        def hyperopt_objective_fn(params):
            params["max_depth"] = int(params["max_depth"])

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

            return {"status": STATUS_OK, "loss": 1 - best_score, "metrics": metrics}

        return hyperopt_objective_fn
