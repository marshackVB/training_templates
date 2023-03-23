
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType


def sample_pandas_dataframe(dir=None):

    dtype = {"PassengerId": "object",
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
             "Survived": "int32"}
    
    if not dir:
        dir = "./tests/data/sample_features.csv"
    
    sample_dataframe = pd.read_csv(dir, header=0, dtype=dtype)
    return sample_dataframe


def sample_spark_dataframe(dir=None):

    spark = SparkSession.builder.getOrCreate()
    sample_pandas_df = sample_pandas_dataframe(dir)

    spark_schema = StructType([StructField("PassengerId", StringType(), True),
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
                               StructField("Survived", IntegerType(), True)])
    
    sample_dataframe = spark.createDataFrame(sample_pandas_df, schema=spark_schema)
    return sample_dataframe