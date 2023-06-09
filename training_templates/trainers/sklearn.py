from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from training_templates.data_utils import train_val_split
from training_templates.mlflow import mlflow_experiment, MLflowExperimentMixin
from training_templates.data_utils import join_train_val_features


class SkLearnPipelineBase(ABC):
    """
    This class is intended to be inherited by child classes. It implements a typical sci-kit learn data processing workflow that can
    be integrated into model training classes.

    Arguments:

        model: A scikit-learn compatibale model.

        model_name: A string representation of the model name; used for MLflow Experiment logging.

        model_type: One of 'classifier' or 'regressor'. This is used by the mlflow.evaluate() method to calculated
                    statistics on the evaluation dataset.

        preprocessing_pipeline: A scikit-learn ColumnTransformer that handles all data preprocessing.

        label_col: The name of the label/target column.

        random_state: And integer to used for initializing methods that perform operations such as train/test split.

        train_eval_shuffle: Indication of data should be shuffled before spliting features into training and evaluation datasets.

        commit_hash: The commit hash of the code version training the model; this is intended to be assigned programatically.

        release_version: The release version of othe code training the model; this is intended to be assigned programatically.
    """

    def __init__(
        self,
        *,
        model: Callable,
        model_name: str,
        model_type: str,
        preprocessing_pipeline: ColumnTransformer,
        label_col: str,
        random_state: int = 123,
        train_eval_shuffle: bool = True,
        commit_hash: str = "",
        release_version: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.preprocessing_pipeline = preprocessing_pipeline
        self.label_col = label_col
        self.random_state = random_state
        self.train_eval_shuffle = train_eval_shuffle
        self.commit_hash = commit_hash
        self.release_version = release_version

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

    @abstractmethod
    def train(self) -> None:
        """
        The model training code.
        """


class SkLearnHyperoptTrainer(SkLearnPipelineBase, MLflowExperimentMixin):
    """
    A scikit-learn pipeline based model training workflow that tunes hyperparameters
    using hyperopt.

    Arguments:
       delta_feature_table: The name of the Delta table containing the model features.

       delta_train_val_id_table: The namne of the Delta table that contains the column of observations primary keys
                                 that make up the training and evaluation datasets.

       train_size: The proportion of observations held for the training dataset; the remaining are used for
                   the validation dataset.

       tune: A tuner class that handles hyperparameter tuning.

       mlflow_experiment_location: Mflow experiment location where run will be logged.

       mlflow_run_description: A desciption to log with the MLflow run.

    """

    def __init__(
        self,
        *,
        delta_feature_table: str,
        delta_train_val_id_table: str,
        train_size: float,
        tuner,
        mlflow_experiment_location: str,
        mlflow_run_description: str = "",
        **kwargs,
    ):
        self.delta_feature_table = delta_feature_table
        self.delta_train_val_id_table = delta_train_val_id_table
        self.train_size = train_size
        self.feature_df = join_train_val_features(
            self.delta_feature_table, self.delta_train_val_id_table
        )
        self.tuner = tuner
        self.mlflow_experiment_location = mlflow_experiment_location
        self.mlflow_run_description = mlflow_run_description
        super().__init__(**kwargs)

    @mlflow_experiment
    def train(self) -> None:
        print("Splitting features into training and validation datasets")
        self.X_train, self.X_val, self.y_train, self.y_val = train_val_split(
            self.feature_df,
            self.label_col,
            self.train_size,
            self.train_eval_shuffle,
            self.random_state,
        )

        self.X_train_transformed = self.preprocessing_pipeline.fit_transform(
            self.X_train
        )
        self.X_val_transformed = self.preprocessing_pipeline.transform(self.X_val)

        self.model_params = self.tuner.tune(
            self.init_model,
            self.X_train_transformed,
            self.X_val_transformed,
            self.y_train,
            self.y_val,
            self.random_state,
        )

        print("\nTraining model with best hyperparameters")
        self.model_pipeline = self.init_model_pipeline(self.model_params)
        self.model_pipeline.fit(self.X_train, self.y_train)
