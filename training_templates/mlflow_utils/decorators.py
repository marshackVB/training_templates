import functools

import mlflow
import pandas as pd

from training_templates.utils import get_commit_info

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

            logging_attributes = {}
            for attribute, value in self.__dict__.items():
                if attribute not in self.__class__.logging_attributes_to_exclude:
                    logging_attributes[attribute] = value

            mlflow.log_dict(logging_attributes, "class_instance_attributes.json")

            print(
                f"Training complete - run id: {self.run_id}, experiment: {self.mlflow_experiment_location}"
            )
        
    return wrapper