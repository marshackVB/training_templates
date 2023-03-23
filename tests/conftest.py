import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import logging
import mlflow
from training_templates.tests.data_loading_utils import sample_spark_dataframe
from training_templates.training_templates.data_utils import train_test_split



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


@pytest.fixture(scope="session")
def feature_table():
    spark = SparkSession.builder.getOrCreate()

    df = sample_spark_dataframe()

    feature_table_name = 'default.feature_table'
    df.write.mode('overwrite').format('delta').saveAsTable(feature_table_name)

    train_test_split(feature_table_name, unique_id='PassengerId', train_val_size=0.8, allow_overwrite=True)

    yield 'default.feature_table'


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
