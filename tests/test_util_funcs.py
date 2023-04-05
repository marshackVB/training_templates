import pandas as pd
import pytest

from training_templates.data_utils import train_test_split, sample_spark_dataframe


@pytest.mark.parametrize("train_val_size", [0.1, 0.5, 0.8])
def test_train_test_split_proportions(spark, train_val_size):
    spark_df = sample_spark_dataframe()

    table_name = "default.test_case"
    spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

    train, test = train_test_split(
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
    with pytest.raises(Exception) as e:
        spark_df = sample_spark_dataframe(spark)

        table_name = "default.test_case"
        spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

        train, test = train_test_split(
            feature_table_name=table_name,
            unique_id="PassengerId",
            train_val_size=0.8,
            allow_overwrite=True,
        )

        train, test = train_test_split(
            feature_table_name=table_name, unique_id="PassengerId", train_val_size=0.8
        )


def test_train_test_split_allow_overwrite(spark):
    spark_df = sample_spark_dataframe()

    table_name = "default.test_case"
    spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

    starting_train_size = 0.8
    train, test = train_test_split(
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
    train, test = train_test_split(
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
    Test that both the training and testing datasets contain only
    one column and that column is the primary key column
    """

    spark_df = sample_spark_dataframe()

    table_name = "default.test_case"
    spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

    starting_train_size = 0.8
    primary_key = "PassengerId"
    train, test = train_test_split(
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
