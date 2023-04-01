import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union

import mlflow
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, tpe
from hyperopt.early_stop import no_progress_loss
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from training_templates.utils import get_commit_info


class SkLearnPipelineABC(ABC):
    """
    This abstract class defines a workflow for scikit-learn based training.
    """

    @abstractmethod
    def init_preprocessing_pipeline(self):
        """
        Return a scikit-learn ColumnTransformers
        """

    @abstractmethod
    def init_model(self, params):
        """
        Return an instance of the model initialized with
        a set of hyper-parameters
        """

    @abstractmethod
    def init_model_pipeline(self, model_params):
        """
        Combine the ColumnTransformer with the model into
        a scikit-learn Pipeline
        """

    @abstractmethod
    def train_test_split(self):
        """
        Split a Pandas DataFrame of features into training and evaluation
        datasets
        """

    @abstractmethod
    def tune_hyperparameters(self):
        """
        Run a hyper-parameter search, such as a Hyperopt trials workflow
        to find the best model parameters. Save these parameters as an
        instance attribute that can be referenced by the train method
        """

    @abstractmethod
    def train(self):
        """
        Call the full training workflow, logging all information and
        artifacts to an MLflow experiment
        """


class SkLearnHyperoptBase(SkLearnPipelineABC):
    """
    This class it intended to be inherited by child classes. It implements a standard workflow to train
    a scikit-learn model along with MLflow logging. It performs the following steps:
         - Splits features into training/evaluation datasets.
         - Creates a data processing and and model training scikit-learn Pipeline.
         - Searches through a provided hyperparameter space using Hyperopt and returns the best hyperparameters.
         - Trains a final model with the best hyperparameters.
         - Logs the trained model as well as training and evaluation fit statistics to an MLflow experiment run.

    Note:
        This class has one method that is intended to be overridden, config_hyperopt_objective_fn. This method should return
        a function that represents the Hyperopt objective function. The way this function is specified can differ depending
        on the model, thus, it is not defined in this class.

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
        hyerparameter_space: A dictionary contain the parameter names and hyperopt ranges that define the parameter search space.
        hyperopt_max_evals: The maximum number of hyperopt experiment runs allowed.
        hyperopt_iteration_stop_count: Use for early stopping; the maximum experiment allowed before improvement in the validation criteria (the loss
                                       returned by the hyperopt objective function).
        hyperopt_early_stopping_threshold: The percentage increase in the hyperopt loss metric that must occure to precent early stopping.
        mlflow_experiment_location: The directory of the MLflow experiment.
        mlflow_run_description: A text description to log to an MLflow run.
        random_state: And integer to used for initializing methods that perform operations such as train/test split.
        train_eval_shuffle: Indication of data should be shuffled before spliting features into training and evaluation datasets
        commit_hash: The commit hash of the code version training the model; this is intended to be assigned programatically.
        release_version: The release version of othe code training the model; this is intended to be assigned programatically.
    """

    def __init__(
        self,
        model: Callable,
        model_name: str,
        delta_feature_table: str,
        delta_train_val_id_table: str,
        train_size: float,
        numerical_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str],
        label_col: str,
        problem_type: str,
        hyperparameter_space: Dict[str, Any],
        hyperopt_max_evals: int,
        hyperopt_iteration_stop_count: int,
        hyperopt_early_stopping_threshold: float,
        mlflow_experiment_location: str,
        mlflow_run_description: str,
        random_state: int = 123,
        train_eval_shuffle: bool = True,
        commit_hash: str = "",
        release_version: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.delta_feature_table = delta_feature_table
        self.delta_train_val_id_table = delta_train_val_id_table
        self.feature_df = self._join_train_val_features()
        self.train_size = train_size
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.all_feature_cols = [*numerical_cols, *categorical_cols, *binary_cols]
        self.label_col = label_col
        self.problem_type = problem_type
        self.hyperparameter_space = hyperparameter_space
        self.hyperopt_max_evals = hyperopt_max_evals
        self.hyperopt_iteration_stop_count = hyperopt_iteration_stop_count
        self.hyperopt_early_stopping_threshold = hyperopt_early_stopping_threshold
        self.mlflow_experiment_location = mlflow_experiment_location
        self.mlflow_run_description = mlflow_run_description
        self.random_state = random_state
        self.train_eval_shuffle = train_eval_shuffle
        self.commit_hash = commit_hash
        self.release_version = release_version
        self.logging_attributes_to_exclude = [
            "feature_df",
            "model",
            "X_train",
            "X_val",
            "y_train",
            "y_val",
            "hyperparameter_space",
        ]

    def _join_train_val_features(self):
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

    def init_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Configure and return a scikit-learn ColunTransformer to handle all data preprocessing and
        encoding.
        """
        binary_transform = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing")
        )
        numeric_transform = make_pipeline(SimpleImputer(strategy="most_frequent"))
        categorical_transform = make_pipeline(
            SimpleImputer(
                missing_values=None, strategy="constant", fill_value="missing"
            ),
            OneHotEncoder(handle_unknown="ignore"),
        )

        preprocessing_pipeline = ColumnTransformer(
            [
                ("categorical_cols", categorical_transform, self.categorical_cols),
                ("numerical_cols", numeric_transform, self.numerical_cols),
                ("binary_cols", binary_transform, self.binary_cols),
            ],
            remainder="drop",
        )

        return preprocessing_pipeline

    def init_model(self, model_params: Dict[str, Any]) -> Callable:
        """
        Create a function that returns an instance of the model
        while passing in the hyperparamter values
        """
        return self.model(**model_params)

    def init_model_pipeline(self, model_params: Dict[str, Any]) -> Pipeline:
        """
        Combines the proprocessing pipeline and the model
        into a scikit-learn Pipeline and returns the pipeline
        """
        preprocessing_pipeline = self.init_preprocessing_pipeline()
        model = self.init_model(model_params)
        classification_pipeline = Pipeline(
            [("preprocessing_pipeline", preprocessing_pipeline), ("model", model)]
        )
        return classification_pipeline

    def train_test_split(self) -> None:
        """
        Split a Pandas DataFrame of features into training and validation datasets
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.feature_df[self.all_feature_cols],
            self.feature_df[self.label_col],
            train_size=self.train_size,
            random_state=self.random_state,
            shuffle=self.train_eval_shuffle,
        )

    def config_hyperopt_objective_fn(
        self, X_train_transformed, X_val_transformed
    ) -> Callable:
        """
        Return a Hyperopt object function for the type of model to be trained
        """
        raise NotImplementedError(
            "You must override this method with a Hyperopt ojective function for your model"
        )

    def transform_features_for_hyperopt(self):
        """
        Transform the training and evaluation features; These feature are passed
        to the Hyperopt objective function to eliminated repeated feature calculations
        during each experiment.
        """
        preprocessing_pipeline = self.init_preprocessing_pipeline()
        X_train_transformed = preprocessing_pipeline.fit_transform(self.X_train)
        X_val_transformed = preprocessing_pipeline.transform(self.X_val)

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

        final_model_parameters: Dict[str, Union[str, int, float]] = {}

        for parameter, value in best_params.items():
            if parameter in ["n_estimators", "max_depth"]:
                final_model_parameters[parameter] = int(value)
            else:
                final_model_parameters[parameter] = value

        print("Best model parameters:")
        for param, value in best_params.items():
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
            instance_attributes = copy.deepcopy(self.__dict__)
            for attribute in self.logging_attributes_to_exclude:
                instance_attributes.pop(attribute)

            mlflow.log_dict(instance_attributes, "class_instance_attributes.json")

            print(
                f"Training complete - run id: {self.run_id}, experiment: {self.mlflow_experiment_location}"
            )
