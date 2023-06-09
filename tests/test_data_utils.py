import pandas as pd
import pytest

from training_templates.data_utils import join_train_val_features, sample_spark_dataframe, spark_train_test_split


def test_join_train_val_features(spark, default_feature_table):
    """
    Spark DataFrames of feature values and test/val record ids are
    joined and converted to a Pandas DataFrame.
    """
    joined_features = join_train_val_features(default_feature_table, 
                                              f"{default_feature_table}_train")

    assert isinstance(joined_features, pd.DataFrame)
    assert joined_features.shape[0] > 0


@pytest.mark.parametrize("spark, train_val_size", [("spark", 0.1), 
                                                   ("spark", 0.5), 
                                                   ("spark", 0.8)], indirect=["spark"],
)
def test_train_test_split_proportions(spark, train_val_size):
    """
    Test and train/validation dataset split proportions are correct.
    """
    spark_df = sample_spark_dataframe()

    table_name = "default.test_case"
    spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

    train, test = spark_train_test_split(
            feature_table_name=table_name,
            unique_id="PassengerId",
            train_val_size=train_val_size,
            allow_overwrite=True,
        )

    train_cnt = spark.table(train).count()
    test_cnt = spark.table(test).count()
    actual_train_size = round(train_cnt / (train_cnt + test_cnt), 1)

    assert actual_train_size == train_val_size


def test_train_test_split_forbid_overwrite(spark):
    """
    By default, Delta tables for test record ids and train/validation record ids
    cannot be overwritten.
    """
    with pytest.raises(Exception) as e:
        spark_df = sample_spark_dataframe(spark)

        table_name = "default.test_case"
        spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

        train, test = spark_train_test_split(
            feature_table_name=table_name,
            unique_id="PassengerId",
            train_val_size=0.8,
            allow_overwrite=True,
        )

        train, test = spark_train_test_split(
            feature_table_name=table_name, unique_id="PassengerId", train_val_size=0.8
        )


def test_train_test_split_allow_overwrite(spark):
    """
    By changing the allow_overwrite setting to True, Delta tables for test
    record ids and train/validation record ids cannot be overwritten.
    """
    spark_df = sample_spark_dataframe()

    table_name = "default.test_case"
    spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

    starting_train_size = 0.8
    train, test = spark_train_test_split(
        feature_table_name=table_name,
        unique_id="PassengerId",
        train_val_size=starting_train_size,
        allow_overwrite=True,
    )

    train_cnt = spark.table(train).count()
    test_cnt = spark.table(test).count()
    actual_train_size = round(train_cnt / (train_cnt + test_cnt), 1)
    assert actual_train_size == starting_train_size

    ending_train_size = 0.8
    train, test = spark_train_test_split(
        feature_table_name=table_name,
        unique_id="PassengerId",
        train_val_size=ending_train_size,
        allow_overwrite=True,
    )

    train_cnt = spark.table(train).count()
    test_cnt = spark.table(test).count()
    actual_train_size = round(train_cnt / (train_cnt + test_cnt), 1)
    assert actual_train_size == ending_train_size


def test_train_test_split_single_key_col(spark):
    """
    The Delta tables for testing and train/validation records
    contain only one column (distinct id column).
    """
    spark_df = sample_spark_dataframe()

    table_name = "default.test_case"
    spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

    starting_train_size = 0.8
    primary_key = "PassengerId"
    train, test = spark_train_test_split(
        feature_table_name=table_name,
        unique_id=primary_key,
        train_val_size=starting_train_size,
        allow_overwrite=True,
    )

    train_col = spark.table(train).columns
    assert len(train_col) == 1
    assert train_col[0] == primary_key

    test_col = spark.table(test).columns
    assert len(test_col) == 1
    assert test_col[0] == primary_key
