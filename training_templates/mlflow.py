import functools
from typing import Dict, Any, Union

import mlflow
import pandas as pd


def get_or_create_experiment(experiment_location: str) -> None:
    """
    Given a DBFS directory, create an MLflow Experiment using that location if one
    does not already exist
    """

    if not mlflow.get_experiment_by_name(experiment_location):
        print("Experiment does not exist. Creating experiment")

        mlflow.create_experiment(experiment_location)

    mlflow.set_experiment(experiment_location)


def get_serializable_attributes(class_dict: Dict[str, Any]) -> Dict[str, Union[int, str, bool, float]]:
    serializable_attributes = {}
    serializable_attribute_types = [int, str, bool, float]
    for attribute_name, attribute_value  in class_dict.items():
        if type(attribute_value) in serializable_attribute_types:
            serializable_attributes[attribute_name] = attribute_value
    return serializable_attributes


def get_commit_info(commit_sha, release_version):
    tags = {"commit_sha": commit_sha, "release_version": release_version}
    return tags


def mlflow_disable_autolog(training_func):
    """
    A Decorater that initializes an MLflow context to log information 
    to an exising Experiment. This function is designed to be applied 
    to class methods and expect the following instance attributes to 
    exist.

    self.mlflow_experiment_location: (str) The MLflow Experiment location.
    self.run_id: (str) The run id of the experiment to access.

    """
    @functools.wraps(training_func)
    def wrapper(self, *args, **kwargs):
        mlflow.autolog(disable=True)
        training_func(self, *args, **kwargs)
        mlflow.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                silent=True,
            )
    return wrapper


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


def mlflow_hyperopt_experiment(training_func):
    """
    This decorator performs MLflow Experiment initialization and model evaluation. It is 
    designed to be applied to a class' train methods. Additional decorator functions can
    be created to implement different MLflow logging workflows.
     
    The instance and class attributes listed below are expected to exist when this decorator
    is applied to an instance method

    self.mlflow_run_description: (str) Desciption of the MLflow Experiment run.
    self.mlflow_experiment_location: (str) The MLflow Experiment location.
    self.model_name: (str) The name of the model, used to name the run.
    self.model_type: (str) The type of model (classifier or regressor).
    self.label_col: (str) The name of the label column to predict.
    self.X_val: (pd.DataFrame) The validation feature table.
    self.y_val: (pd.DataFrame) The validation label column table.
    self.commit_hash: (str) A commit hash associated with the code.
    self.release_version: (str) A github release version.
    self.__class__.logging_attributes_to_exclude: (List[str]) Columns that cannot be serialized to JSON
        and therefore cannot be logged as an MLflow artifact.

    Example:
        class ModelTrainer(ParentTrainer):
            def __init__(*args, **kwargs):
                super().__init__(*args, **kwargs)

            @mlflowexperiment
            def train(self):
                pass
    """
    @functools.wraps(training_func)
    def wrapper(self, *args, **kwargs):

        mlflow.set_experiment(self.mlflow_experiment_location)

        tags = {"class_name": self.__class__.__name__}

        with mlflow.start_run(run_name=self.model_name, tags=tags, description=self.mlflow_run_description) as run:
            self.run_id = run.info.run_id

            training_func(self, *args, **kwargs)

            if self.commit_hash:
                tags = get_commit_info(self.commit_hash, self.release_version)
                mlflow.set_tags(tags)

            self.model_uri = f"runs:/{self.run_id}/model"

            eval_features_and_labels = pd.concat([self.X_val, self.y_val], axis=1)

            print("Scoring validation dataset; logging metrics and artifacts")
     
            mlflow.evaluate(
                        self.model_uri,
                        data=eval_features_and_labels,
                        targets=self.label_col,
                        model_type=self.model_type,
                    )

            attributes_to_log = get_serializable_attributes(self.__dict__)
            mlflow.log_dict(attributes_to_log, "class_instance_attributes.json")

            print(
                f"Training complete - run id: {self.run_id}, experiment: {self.mlflow_experiment_location}"
            )
        
    return wrapper


class MLflowExperimentMixin:
    """
    A collection of functions to log data to existing MLflow Experiment runs.
    By inheriting this class, a child class is able to use these methods
    to easily log information to its latest Experiment run, without the need
    to pass any Experiment location or run information manually.
    """
    def _log_response(self, logging_type: str):
        print(f"{logging_type} logged to run: {self.run_id}, experiment: {self.mlflow_experiment_location}")

    @mlflow_logger
    def log_params(self, params):
        mlflow.log_params(params)
        self._log_response("Parameters")

    @mlflow_logger
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
        self._log_response("Metrics")

    @mlflow_logger
    def set_tags(self, tags):
        mlflow.set_tags(tags)
        self._log_response("Tags")

    @mlflow_logger
    def log_artifacts(self, artifacts):
        mlflow.log_artifacts(artifacts)
        self._log_response("Artifacts")