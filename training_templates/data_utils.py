import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def train_test_split(
    feature_table_name: str,
    unique_id: str,
    train_val_size: float,
    random_seed: int = 123,
    allow_overwrite: bool = False,
):
    """
    Randomly split features table intro training and testing DataFrames. Write the unique record ids of each DataFrame to
    a Delta Table. Feature values can then be joined to these tables. The training table can be further split into training
    and validation datasets for hyper-parameter tuning or can be used directly in the case of k-fold cross validation.
    """

    spark = SparkSession.builder.getOrCreate()

    test_size = 1 - train_val_size
    splits_df = (
        spark.table(feature_table_name)
        .select(unique_id)
        .randomSplit(weights=[train_val_size, test_size], seed=random_seed)
    )

    train_df = splits_df[0]
    test_df = splits_df[1]

    train_output_table_name = f"{feature_table_name}_train"
    test_output_table_name = f"{feature_table_name}_test"

    if not allow_overwrite:
        train_record_cnt = (
            spark.table(train_output_table_name).count()
            if spark.catalog.tableExists(test_output_table_name)
            else 0
        )
        test_record_cnt = (
            spark.table(test_output_table_name).count()
            if spark.catalog.tableExists(test_output_table_name)
            else 0
        )

        if train_record_cnt > 0 or test_record_cnt > 0:
            raise Exception(
                f"Train and test tables already exist: train record count: {train_record_cnt} and test table record: {test_record_cnt}"
            )

    train_df.write.mode("overwrite").format("delta").saveAsTable(
        train_output_table_name
    )
    test_df.write.mode("overwrite").format("delta").saveAsTable(test_output_table_name)

    print(
        f"Training record ids written to {train_output_table_name}, Testing ids written to {test_output_table_name}"
    )

    return (train_output_table_name, test_output_table_name)



def join_train_val_features(delta_feature_table: str, delta_train_val_id_table: str) -> pd.DataFrame:
    """
    Join the feature table to the ids associated with the training and evaluation observations.
    Remaining observations in the feature table are set aside as a test dataset.
    """
    spark = SparkSession.builder.getOrCreate()

    features = spark.table(delta_feature_table)
    train_val_ids = spark.table(delta_train_val_id_table)
    train_val_ids_primary_key = train_val_ids.columns[0]

    train_val_features = features.join(
        train_val_ids, [train_val_ids_primary_key], "inner"
    )

    return train_val_features.toPandas()


def sample_pandas_dataframe():
    dtype = {
        "PassengerId": "object",
        "Fare": "float64",
        "Embarked": "object",
        "Pclass": "object",
        "Parch": "object",
        "TicketChars": "object",
        "CabinChar": "object",
        "CabinMulti": "object",
        "FareRounded": "float64",
        "Name": "object",
        "Sex": "object",
        "Age": "float64",
        "SibSp": "object",
        "NamePrefix": "object",
        "NameSecondary": "object",
        "NameMultiple": "object",
        "Survived": "int32",
    }

    current_dir = os.path.dirname(__file__)
    data_path = f"{current_dir}/data/sample_features.csv"

    sample_dataframe = pd.read_csv(data_path, header=0, dtype=dtype)
    return sample_dataframe


def sample_spark_dataframe():
    spark = SparkSession.builder.getOrCreate()
    sample_pandas_df = sample_pandas_dataframe()

    spark_schema = StructType(
        [
            StructField("PassengerId", StringType(), True),
            StructField("Fare", FloatType(), True),
            StructField("Embarked", StringType(), True),
            StructField("Pclass", StringType(), True),
            StructField("Parch", StringType(), True),
            StructField("TicketChars", StringType(), True),
            StructField("CabinChar", StringType(), True),
            StructField("CabinMulti", StringType(), True),
            StructField("FareRounded", FloatType(), True),
            StructField("Name", StringType(), True),
            StructField("Sex", StringType(), True),
            StructField("Age", FloatType(), True),
            StructField("SibSp", StringType(), True),
            StructField("NamePrefix", StringType(), True),
            StructField("NameSecondary", StringType(), True),
            StructField("NameMultiple", StringType(), True),
            StructField("Survived", IntegerType(), True),
        ]
    )

    sample_dataframe = spark.createDataFrame(sample_pandas_df, schema=spark_schema)
    return sample_dataframe
