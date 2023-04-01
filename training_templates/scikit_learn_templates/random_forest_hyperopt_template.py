from collections import OrderedDict

from hyperopt import STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from training_templates.scikit_learn_templates.base_templates import SkLearnHyperoptBase


class RandomForeastHyperoptTrainer(SkLearnHyperoptBase):
    """
    Implements a Hyperopt objective function for a Random Forest training workflow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(RandomForestClassifier, "random_forest", *args, **kwargs)

    def config_hyperopt_objective_fn(self, X_train_transformed, X_val_transformed):
        """
        Return an Hyperopt objective function for an Random Forest model.
        """

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
