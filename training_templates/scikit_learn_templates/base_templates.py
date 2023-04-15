from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union, Tuple

import mlflow
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, tpe, space_eval
from hyperopt.early_stop import no_progress_loss
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from training_templates.utils import get_commit_info


class SkLearnPipelineBase(ABC):
    """
    This class is intended to be inherited by child classes. It implements a typical sci-kit learn data processing workflow that can
    be integrated into model training classes. It performs the following steps:
        - Splits features into training/evaluation datasets.
        - Creates a data processing and and model training scikit-learn Pipeline.

    Arguments:
        delta_feature_table: The name of the Delta table containing the model features.

        delta_train_val_id_table: The namne of the Delta table that contains the column of observations primary keys
                                that make up the training and evaluation datasets.

        numerical_cols: A list of numerical feature column names.

        categorical_cols: A list of categorical feature column names.

        binary_ols: A list of binary feature column names.

        label_col: The name of the label/target column.

        problem_type: One of 'classification' or 'regression'. This is used by the mlflow.evaluate() method to calculated
                      statistics on the evaluation dataset.

        random_state: And integer to used for initializing methods that perform operations such as train/test split.

        train_eval_shuffle: Indication of data should be shuffled before spliting features into training and evaluation datasets.

        commit_hash: The commit hash of the code version training the model; this is intended to be assigned programatically.

        release_version: The release version of othe code training the model; this is intended to be assigned programatically.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        model: Callable,
        model_name: str,
        problem_type: str,
        preprocessing_pipeline: ColumnTransformer,
        delta_feature_table: str,
        delta_train_val_id_table: str,
        train_size: float,
        label_col: str,
        random_state: int = 123,
        train_eval_shuffle: bool = True,
        commit_hash: str = "",
        release_version: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.problem_type = problem_type
        self.preprocessing_pipeline = preprocessing_pipeline
        self.delta_feature_table = delta_feature_table
        self.delta_train_val_id_table = delta_train_val_id_table
        self.feature_df = self._join_train_val_features()
        self.train_size = train_size
        self.label_col = label_col
        self.random_state = random_state
        self.train_eval_shuffle = train_eval_shuffle
        self.commit_hash = commit_hash
        self.release_version = release_version

    def _join_train_val_features(self) -> pd.DataFrame:
        """
        Join the feature table to the ids associated with the training and evaluation observations.
        Remaining observations in the feature table are set aside as a test dataset.
        """
        spark = SparkSession.builder.getOrCreate()

        features = spark.table(self.delta_feature_table)
        train_val_ids = spark.table(self.delta_train_val_id_table)
        train_val_ids_primary_key = train_val_ids.columns[0]

        train_val_features = features.join(
            train_val_ids, [train_val_ids_primary_key], "inner"
        )

        return train_val_features.toPandas()

    def train_test_split(self) -> None:
        """
        Split a Pandas DataFrame of features into training and validation datasets
        """
        non_label_cols = [
            col for col in self.feature_df.columns if col != self.label_col
        ]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.feature_df[non_label_cols],
            self.feature_df[self.label_col],
            train_size=self.train_size,
            random_state=self.random_state,
            shuffle=self.train_eval_shuffle,
        )

    def init_model(
        self, model_params: Union[Dict[str, Union[str, int, float]], None] = None
    ) -> Callable:
        """
        Create a function that returns an instance of the model
        while passing in the hyperparamter values
        """
        if not model_params:
            return self.model()
        else:
            return self.model(**model_params)

    def init_model_pipeline(self, model_params: Dict[str, Any]) -> Pipeline:
        """
        Combines the proprocessing pipeline and the model
        into a scikit-learn Pipeline and returns the pipeline
        """
        model = self.init_model(model_params)
        classification_pipeline = Pipeline(
            [("preprocessing_pipeline", self.preprocessing_pipeline), ("model", model)]
        )
        return classification_pipeline


class SkLearnHyperoptBase(SkLearnPipelineBase, ABC):
    """
    This class it intended to be inherited by child classes. It implements a standard workflow to train
    a scikit-learn model with parameter tuning via Hyperopt and MLflow logging. It performs the following steps:
        - Searches through a provided hyperparameter space using Hyperopt and returns the best hyperparameters
          for a model.
        - Trains a final model with the best hyperparameters.
        - Logs the trained model as well as training and evaluation fit statistics to an MLflow experiment run.

    Arguments:
        hyerparameter_space: A dictionary contain the parameter names and hyperopt ranges that define the parameter search space.

        hyperopt_max_evals: The maximum number of hyperopt experiment runs allowed.

        hyperopt_iteration_stop_count: Use for early stopping; the maximum experiment allowed before improvement in the validation criteria (the loss
                                       returned by the hyperopt objective function).

        hyperopt_early_stopping_threshold: The percentage increase in the hyperopt loss metric that must occure to precent early stopping.

        mlflow_experiment_location: The directory of the MLflow experiment.

        mlflow_run_description: A text description to log to an MLflow run.
    """

    logging_attributes_to_exclude = [
        "feature_df",
        "preprocessing_pipeline",
        "model",
        "X_train",
        "X_val",
        "y_train",
        "y_val",
        "hyperparameter_space",
    ]

    def __init__(
        self,
        *,
        hyperparameter_space: Dict[str, Any],
        hyperopt_max_evals: int,
        hyperopt_iteration_stop_count: int,
        hyperopt_early_stopping_threshold: float,
        mlflow_experiment_location: str,
        mlflow_run_description: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hyperparameter_space = hyperparameter_space
        self.hyperopt_max_evals = hyperopt_max_evals
        self.hyperopt_iteration_stop_count = hyperopt_iteration_stop_count
        self.hyperopt_early_stopping_threshold = hyperopt_early_stopping_threshold
        self.mlflow_experiment_location = mlflow_experiment_location
        self.mlflow_run_description = mlflow_run_description

    def format_hyperopt_for_sklearn(
        self, hyperopt_params: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Union[str, int, float]]:
        """
        Convert floating point parameters to integers when scikit-learn
        expects integers
        """
        default_model = self.model()
        for param, value in hyperopt_params.items():
            defaul_value = default_model.__dict__[param]
            if isinstance(defaul_value, int):
                hyperopt_params[param] = int(hyperopt_params[param])
        return hyperopt_params

    @abstractmethod
    def config_hyperopt_objective_fn(
        self, X_train_transformed: pd.DataFrame, X_val_transformed: pd.DataFrame
    ) -> Callable:
        """
        Return a Hyperopt object function for the type of model to be trained
        """

    def transform_features_for_hyperopt(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform the training and evaluation features; These feature are passed
        to the Hyperopt objective function to eliminated repeated feature calculations
        during each experiment.
        """
        X_train_transformed = self.preprocessing_pipeline.fit_transform(self.X_train)
        X_val_transformed = self.preprocessing_pipeline.transform(self.X_val)

        return (X_train_transformed, X_val_transformed)

    def tune_hyperparameters(self) -> Dict[str, Union[str, int, float]]:
        """
        Launch the Hyperopt Trials workflow
        """

        mlflow.autolog(disable=True)

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
        final_model_parameters = self.format_hyperopt_for_sklearn(best_params)

        print("Best model parameters:")
        for param, value in final_model_parameters.items():
            print(param, value)

        print("\nBest model statistics:")
        for metric, value in trials.best_trial["result"]["metrics"].items():
            print(metric, value)

        return final_model_parameters

    def train(self) -> None:
        """
        Execute the full training workflow.
            - Split features into training and validation datasets
            - Execute hyperopt parameter tuning workflow and return best parameters
            - Set MLflow experiment location
            - Train final models
            - Calculate and log validation statistics
        """
        print("Splitting features into training and validation datasets")
        self.train_test_split()

        print("Searching hyperparameter space")
        self.model_params = self.tune_hyperparameters()

        mlflow.set_experiment(self.mlflow_experiment_location)
        tags = {"class_name": self.__class__.__name__}
        with mlflow.start_run(
            run_name=self.model_name, tags=tags, description=self.mlflow_run_description
        ) as run:
            self.run_id = run.info.run_id

            mlflow.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                silent=True,
            )

            print("\nTraining model with best hyperparameters")
            model_training_pipeline = self.init_model_pipeline(self.model_params)
            model_training_pipeline.fit(self.X_train, self.y_train)

            if self.commit_hash:
                tags = get_commit_info(self.commit_hash, self.release_version)
                mlflow.set_tags(tags)

            logged_model = f"runs:/{self.run_id}/model"

            eval_features_and_labels = pd.concat([self.X_val, self.y_val], axis=1)

            print("Scoring validation dataset; logging metrics and artifacts")
            mlflow.evaluate(
                logged_model,
                data=eval_features_and_labels,
                targets=self.label_col,
                model_type="classifier",
            )

            # Log instance attributes as json file
            logging_attributes = {}
            for attribute, value in self.__dict__.items():
                if attribute not in self.__class__.logging_attributes_to_exclude:
                    logging_attributes[attribute] = value

            mlflow.log_dict(logging_attributes, "class_instance_attributes.json")

            print(
                f"Training complete - run id: {self.run_id}, experiment: {self.mlflow_experiment_location}"
            )
