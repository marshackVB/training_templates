# Databricks notebook source
# MAGIC %md ## Example training routines  
# MAGIC If cloning the library repository into a Databricks Repo, the below code can be run without the need to pip install the library. To execute the project code outside of the Repo, pip install the library.

# COMMAND ----------

# Example pip installation from gemfury
# Private repo - create Databricks Secret and pass is as a cluster environment variable: GEMFURY_TOKEN={{secrets/gemfury/GEMFURY_TOKEN}}
# See documentation: https://docs.databricks.com/security/secrets/secrets.html#reference-a-secret-in-an-environment-variable
# %pip install training-templates --index-url https://$GEMFURY_TOKEN:@pypi.fury.io/marshackvb --extra-index-url https://pypi.org/simple -q
# Public repo
# %pip install training-templates==0.1.0 --index-url https://pypi.fury.io/marshackvb --extra-index-url https://pypi.org/simple -q

# COMMAND ----------

# The requirements file exists so that dependecies can be installed for interactive 
# testing/experimentation in Notebooks. This way, the project dependecies can be installed without
# building the package. Module functions and classes can be edited in VSCode and synced to the Repo
# in real time. Then, Notebook cell runs will capture the updates due to the use of %autoreload magic 
# functions, without the need to detach and reatach the Notebook to the Cluster.
%pip install -r ../requirements.txt -q

# COMMAND ----------

# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
%load_ext autoreload
%autoreload 2

# COMMAND ----------

from hyperopt import hp

from training_templates import SkLearnHyperoptBase, XGBoostHyperoptTrainer, RandomForestHyperoptTrainer
from training_templates.data_utils import sample_spark_dataframe, train_test_split
from training_templates.utils import get_or_create_experiment

# COMMAND ----------

# MAGIC %md #### Load example features

# COMMAND ----------

raw_features_table = 'default.mlc_raw_features'

raw_features = sample_spark_dataframe()
raw_features.write.mode('overwrite').format('delta').saveAsTable(raw_features_table)
display(spark.table(raw_features_table))

# COMMAND ----------

# MAGIC %md #### Separate test records from training and validation records  
# MAGIC The DataFrame containing training and validation records is passed to the model. After hyperparameter tuning and other experimentation is compete, the final model can be applied to the holdout test dataset to calculate final, unbiased fit statistics.

# COMMAND ----------

train_table_name, test_table_name = train_test_split(feature_table_name=raw_features_table,
                                                     unique_id='PassengerId',
                                                     train_val_size=0.85,
                                                     allow_overwrite=False)

# COMMAND ----------

# MAGIC %md View distinct IDs associated with the test and training/validation datasets

# COMMAND ----------

display(spark.table(train_table_name))

# COMMAND ----------

display(spark.table(test_table_name))

# COMMAND ----------

train_record_cnt = spark.table(train_table_name).count()
test_record_cnt = spark.table(test_table_name).count()

print(f"Training/validation records: {train_record_cnt}; Test records: {test_record_cnt}")

# COMMAND ----------

# MAGIC %md #### Configure models

# COMMAND ----------

raw_features_table = 'default.mlc_raw_features'
label_col = 'Survived'
categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
numerical_cols = ['Age', 'FareRounded']
binary_cols = ['NameMultiple']

model_params = {"delta_feature_table": raw_features_table, 
                "delta_train_val_id_table": f"{raw_features_table}_train", 
                "train_size": 0.8, 
                "numerical_cols": numerical_cols, 
                "categorical_cols": categorical_cols, 
                "binary_cols": binary_cols, 
                "label_col": label_col, 
                "problem_type": "classification", 
                "hyperopt_max_evals": 500,
                "hyperopt_iteration_stop_count": 50,
                "hyperopt_early_stopping_threshold": 0.05,
                "mlflow_experiment_location": "/Shared/ml_production_experiment"}

# COMMAND ----------

# MAGIC %md ####Train an XGBoost model and log to MLflow

# COMMAND ----------

description = "A test training run for XGBoostHyperoptTrainer"

hyperparameter_space = {'max_depth': hp.quniform('max_depth', 1, 10, 1),
                        'eval_metric': 'auc',
                        'early_stopping_rounds': 50}

model_params["hyperparameter_space"] = hyperparameter_space
model_params["mlflow_run_description"] = description

model = XGBoostHyperoptTrainer(**model_params)

model.train()

# COMMAND ----------

# MAGIC %md ####Train a Random Forest model and log to MLflow

# COMMAND ----------

description = "A test training run for RandomForestHyperoptTrainer"

hyperparameter_space = {"n_estimators": hp.quniform("n_estimators", 20, 1000, 1),
                        "max_features": hp.uniform("max_features", 0.5, 1.0),
                        "criterion": hp.choice("criterion", ["gini", "entropy"])}

model_params["hyperparameter_space"] = hyperparameter_space
model_params["mlflow_run_description"] = description

model = RandomForestHyperoptTrainer(**model_params)

model.train()
