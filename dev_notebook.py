# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from hyperopt import hp

from training_templates.sklearn_templates import SkLearnHyperoptBase, XGBoostHyperoptTrainer, RandomForeastHyperoptTrainer
from training_templates.data_utils import train_test_split
from tests.data_loading_utils import sample_spark_dataframe

# COMMAND ----------

dbutils.fs.ls('dbfs:/Users/marshall.carter@databricks.com')

# COMMAND ----------

df = sample_spark_dataframe('/dbfs/Users/marshall.carter@databricks.com')
display(df)

# COMMAND ----------

df = spark.table('default.passenger_featurs_combined').toPandas()

# COMMAND ----------

df = spark.table('default.passenger_featurs_combined').toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

df.to_csv('sample_features.csv', header=True, index=False)

# COMMAND ----------

df.to_csv('/dbfs/Users/marshall.carter@databricks.com/sample_features.csv', header=True, index=False)

# COMMAND ----------

"""
train_table_name, test_table_name = train_test_split(feature_table_name="default.passenger_featurs_combined",
                                                     unique_id='PassengerId',
                                                     train_val_size=0.1,
                                                     allow_overwrite=False)
"""                                             

# COMMAND ----------

delta_feature_table = "default.passenger_featurs_combined",

label_col = 'Survived'
categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
numerical_cols = ['Age', 'FareRounded']
binary_cols = ['NameMultiple']

# COMMAND ----------

description = "A test training run"

label_col = 'Survived'
categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
numerical_cols = ['Age', 'FareRounded']
binary_cols = ['NameMultiple']

hyperparameter_space = {'n_estimators': hp.quniform('n_estimators', 3, 18, 1)}

model = SkLearnHyperoptBase(delta_feature_table = "default.mlc_test_features", 
                            delta_train_val_id_table = "default.mlc_test_features_train", 
                            train_size = 0.8, 
                            numerical_cols = numerical_cols, 
                            categorical_cols = categorical_cols, 
                            binary_cols = binary_cols, 
                            label_col = label_col, 
                            problem_type = "classification", 
                            hyperparameter_space = hyperparameter_space, 
                            max_evals = 5,
                            experiment_location = "/Shared/ml_production_experiment",
                            description = description, 
                            random_state = 123, 
                            shuffle = True, 
                            commit_hash = None, 
                            release_version = None)

# COMMAND ----------

description = "A test training run for XGBoostHyperoptTrainer"

label_col = 'Survived'
categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
numerical_cols = ['Age', 'FareRounded']
binary_cols = ['NameMultiple']

hyperparameter_space = {'max_depth': hp.quniform('n_estimators', 1, 10, 1),
                        'eval_metric': 'auc',
                        'early_stopping_rounds': 50}

model = XGBoostHyperoptTrainer(delta_feature_table = "default.passenger_featurs_combined", 
                               delta_train_val_id_table = "default.passenger_featurs_combined_train", 
                               train_size = 0.8, 
                               numerical_cols = numerical_cols, 
                               categorical_cols = categorical_cols, 
                               binary_cols = binary_cols, 
                               label_col = label_col, 
                               problem_type = "classification", 
                               hyperparameter_space = hyperparameter_space, 
                               hyperopt_max_evals = 500,
                               hyperopt_iteration_stop_count = 50,
                               hyperopt_early_stopping_threshold = 0.05,
                               mlflow_experiment_location = "/Shared/ml_production_experiment",
                               mlflow_run_description = description, 
                               random_state = 123, 
                               train_eval_shuffle = True, 
                               commit_hash = None, 
                               release_version = None)

model.train()
