import functools

import mlflow


def mlflow_logger(logging_func):
    """
    A Decorater that initializes an MLflow context to log information 
    to an exising Experiment. This function is designed to be applied 
    to class methods and expect the following instance attributes to 
    exist.

    self.mlflow_experiment_location: (str) The MLflow Experiment location.
    self.run_id: (str) The run id of the experiment to access.

    """
    @functools.wraps(logging_func)
    def wrapper(self, *args, **kwargs):
      mlflow.set_experiment(self.mlflow_experiment_location)
      with mlflow.start_run(run_id = self.run_id) as run:
        logging_func(self, *args, **kwargs)
    return wrapper
  

class MLflowExperimentMixin:
    """
    A collection of functions to log data to existing MLflow Experiment runs.
    By inheriting this class, a child class is able to use these methods
    to easily log information to its latest Experiment run, without the need
    to pass any Experiment location or run information manually.
    """
    @mlflow_logger
    def log_params(self, params):
        mlflow.log_params(params)

    @mlflow_logger
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)

    @mlflow_logger
    def set_tags(self, tags):
        mlflow.set_tags(tags)

    @mlflow_logger
    def log_artifacts(self, artifacts):
        mlflow.log_artifacts(artifacts)