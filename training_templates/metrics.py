from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from training_templates.constants import BOOSTED_MODELS


def classification_metrics_to_dict(
    precision, recall, f1, best_iteration=None, postfix="val", round_digits=3
):
    """
    Return metrics in an ordered dictionary. Include the best iteration for boosted models if this
    value is passed
    """
    metrics = OrderedDict()
    metrics[f"precision_{postfix}"] = round(precision, round_digits)
    metrics[f"recall_{postfix}"] = round(recall, round_digits)
    metrics[f"f1_{postfix}"] = round(f1, round_digits)

    if best_iteration:
        metrics["best_iteration"] = best_iteration

    return metrics


def xgboost_classification_metrics(model, x, y, average, round_digits):
    """
    Return metrics for an XGBoost classification model
    """
    best_iteration = model.best_iteration
    best_xgboost_rounds = (0, best_iteration + 1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        model.predict(x, iteration_range=best_xgboost_rounds),
        average=average,
    )

    metrics = classification_metrics_to_dict(
        precision, recall, f1, best_iteration, round_digits
    )

    return metrics


def sklearn_classification_metrics(model, x, y, average, round_digits):
    """
    Return metrics for a scikit-learn classification model
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, model.predict(x), average=average
    )

    metrics = classification_metrics_to_dict(precision, recall, f1, round_digits)

    return metrics


def classification_metrics(model, x, y, average="weighted", round_digits=3):
    """
    Return the appropriate classification metrics function for the model type
    """

    is_boosted_model = False

    if type(model) == Pipeline:
        model_type = model.steps[-1][1]
        if model_type in BOOSTED_MODELS:
            is_boosted_model = True

    elif type(model) in BOOSTED_MODELS:
        is_boosted_model = True

    if is_boosted_model:
        metrics = xgboost_classification_metrics(model, x, y, average, round_digits)
    else:
        metrics = sklearn_classification_metrics(model, x, y, average, round_digits)

    return metrics
