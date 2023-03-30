from pyspark.sql import SparkSession


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
