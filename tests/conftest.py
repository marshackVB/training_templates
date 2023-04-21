import logging
import shutil
import tempfile
from pathlib import Path

from delta import configure_spark_with_delta_pip
from hyperopt import STATUS_OK, hp
from hyperopt.pyll.base import scope
import mlflow
import pytest
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from training_templates.data_utils import spark_train_test_split, sample_spark_dataframe
from training_templates.tuners import SkLearnHyperoptTuner, Tuner


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    This fixture provides preconfigured SparkSession with Hive and Delta support.
    After the test session, temporary warehouse directory is deleted.
    :return: SparkSession

    https://stackoverflow.com/questions/7362900/behaviour-of-pythons-yield
    https://stackoverflow.com/questions/35489844/what-does-yield-without-value-do-in-context-manager
    """
    logging.info("Configuring Spark session for testing environment")
    warehouse_dir = tempfile.TemporaryDirectory().name

    _builder = (
        SparkSession.builder.master("local[1]")
        .config("spark.hive.metastore.warehouse.dir", Path(warehouse_dir).as_uri())
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark: SparkSession = configure_spark_with_delta_pip(_builder).getOrCreate()
    logging.info("Spark session configured")
    yield spark

    logging.info("Shutting down Spark session")
    spark.stop()
    if Path(warehouse_dir).exists():
        shutil.rmtree(warehouse_dir)


@pytest.fixture(scope="session", autouse=True)
def mlflow_local():
    """
    This fixture provides local instance of mlflow with support for tracking and registry functions.
    After the test session:
    * temporary storage for tracking and registry is deleted.
    * Active run will be automatically stopped to avoid verbose errors.
    :return: None
    """
    logging.info("Configuring local MLflow instance")
    tracking_uri = tempfile.TemporaryDirectory().name
    registry_uri = f"sqlite:///{tempfile.TemporaryDirectory().name}"

    mlflow.set_tracking_uri(Path(tracking_uri).as_uri())
    mlflow.set_registry_uri(registry_uri)
    logging.info("MLflow instance configured")
    yield None

    mlflow.end_run()

    if Path(tracking_uri).exists():
        shutil.rmtree(tracking_uri)

    if Path(registry_uri).exists():
        Path(registry_uri).unlink()
    logging.info("Test session finished, unrolling the MLflow instance")


@pytest.fixture(scope="session")
def default_feature_table(spark):
    """
    Create a Delta table and split it into training and test datasets
    """
    df = sample_spark_dataframe()

    feature_table_name = "default.feature_table"
    df.write.mode("overwrite").format("delta").saveAsTable(feature_table_name)

    spark_train_test_split(
        feature_table_name,
        unique_id="PassengerId",
        train_val_size=0.8,
        allow_overwrite=True,
    )

    return "default.feature_table"


@pytest.fixture
def default_training_args(default_feature_table):
    """
    Default trainer arguments to use across tests. This is a partial list, additional
    arguments need to be added within tests.
    """
    categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
    numerical_cols = ['Age', 'FareRounded']
    binary_cols = ['NameMultiple']

    numeric_transform = make_pipeline(SimpleImputer(strategy="most_frequent"))

    categorical_transform = make_pipeline(
        SimpleImputer(
            strategy="constant", fill_value="missing"
        ),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessing_pipeline = ColumnTransformer(
        [
            ("categorical_cols", categorical_transform, categorical_cols),
            ("numerical_cols", numeric_transform, numerical_cols),
            ("binary_cols", 'passthrough', binary_cols),
        ],
        remainder="drop",)

    args = {
        "preprocessing_pipeline": preprocessing_pipeline,
        "delta_feature_table": default_feature_table,
        "delta_train_val_id_table": f"{default_feature_table}_train",
        "label_col": "Survived",
        "train_size": 0.8,
        "mlflow_experiment_location": "/Shared/ml_production_experiment",
        "mlflow_run_description": "A Model instance designed for testing",
        "random_state": 123,
        "train_eval_shuffle": True,
        "commit_hash": None,
        "release_version": None,
    }
    return args


@pytest.fixture
def default_tuner_args():
    """
    Default hyperopt setting to use across tests.
    """
    args = {"hyperopt_max_evals": 20,
            "hyperopt_iteration_stop_count": 5,
            "hyperopt_early_stopping_threshold": 0.05}
    return args


@pytest.fixture
def default_tuner(default_tuner_args):
    """
    A hyperoptopt tuner instance for model training tests.
    """
    hyperparameter_space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 2, 10, 1)),
        "max_features": hp.uniform("max_features", 0.5, 1.0),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }

    default_tuner_args["hyperparameter_space"] = hyperparameter_space
    model_name = "random_forest"
    tuner = Tuner.load_tuner(model_name, default_tuner_args)
    return tuner
      






