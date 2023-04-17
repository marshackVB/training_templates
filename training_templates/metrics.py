from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support


def xgboost_classification_metrics(model, x, y, average="weighted", round_digits=3):

    best_iteration = model.best_iteration
    best_xgboost_rounds = (0, best_iteration + 1)
     
    precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
                    y,
                    model.predict(x, iteration_range=best_xgboost_rounds),
                    average=average,
                )

    metrics = OrderedDict()
    metrics["precision_val"] = round(precision_val, round_digits)
    metrics["recall_val"] = round(recall_val, round_digits)
    metrics["f1_val"] = round(f1_val, round_digits)
    metrics["best_iteration"] = best_iteration

    return metrics


def sklearn_classification_metrics(model, x, y, average="weighted", round_digits=3):
     
    precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
                    y, 
                    model.predict(x), 
                    average=average
            )

    metrics = OrderedDict()
    metrics["precision_val"] = round(precision_val, round_digits)
    metrics["recall_val"] = round(recall_val, round_digits)
    metrics["f1_val"] = round(f1_val, round_digits)

    return metrics