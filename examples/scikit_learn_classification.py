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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from training_templates.tuners import XGBoostHyperoptTuner, SkLearnHyperoptTuner
from training_templates import SkLearnHyperoptTrainer
from training_templates.data_utils import sample_spark_dataframe, spark_train_test_split
from training_templates.mlflow import get_or_create_experiment
from training_templates.metrics import xgboost_classification_metrics, sklearn_classification_metrics

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

train_table_name, test_table_name = spark_train_test_split(feature_table_name=raw_features_table,
                                                           unique_id='PassengerId',
                                                           train_val_size=0.85,
                                                           allow_overwrite=True)

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

# MAGIC %md #### Configure feature preprocessing

# COMMAND ----------

categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
numerical_cols = ['Age', 'FareRounded']
binary_cols = ['NameMultiple']

numeric_transform = make_pipeline(SimpleImputer(strategy="most_frequent"))

categorical_transform = make_pipeline(
    SimpleImputer(
        missing_values=None, strategy="constant", fill_value="missing"
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

# COMMAND ----------

# MAGIC %md #### Random Forest training and logging

# COMMAND ----------

# MAGIC %md Configure tuner

# COMMAND ----------

hyperparameter_space = {"n_estimators": hp.quniform("n_estimators", 20, 1000, 1),
                        "max_features": hp.uniform("max_features", 0.5, 1.0),
                        "criterion": hp.choice("criterion", ["gini", "entropy"])}

tuner_args = {"hyperparameter_space": hyperparameter_space,
              "hyperopt_max_evals": 200 ,
              "hyperopt_iteration_stop_count": 20,
              "hyperopt_early_stopping_threshold": 0.05}

tuner = SkLearnHyperoptTuner(**tuner_args)

# COMMAND ----------

# MAGIC %md Tune and train

# COMMAND ----------

raw_features_table = 'default.mlc_raw_features'
label_col = 'Survived'

description = "An example Random Forest model trained with hyerpopt"

training_params = {"model": RandomForestClassifier,
                   "model_name": "random_forecast",
                   "model_type": "classifier",
                   "delta_feature_table": raw_features_table, 
                   "delta_train_val_id_table": f"{raw_features_table}_train", 
                   "train_size": 0.8, 
                   "preprocessing_pipeline": preprocessing_pipeline,
                   "label_col": label_col,
                   "tuner": tuner,
                   "mlflow_experiment_location": "/Shared/ml_production_experiment",
                   "mlflow_run_description": description}

trainer = SkLearnHyperoptTrainer(**training_params)
trainer.train()

# COMMAND ----------

# MAGIC %md Log to trainer's Experiment run

# COMMAND ----------

tags = {"tuner": "hyperopt"}
trainer.set_tags(tags)

# COMMAND ----------

# MAGIC %md Perform prediction from trainer

# COMMAND ----------

predictions = trainer.model_pipeline.predict(trainer.X_train)
predictions[:10]

# COMMAND ----------

# MAGIC %md Generate and log custom metrics

# COMMAND ----------

custom_metrics = sklearn_classification_metrics(model = trainer.model_pipeline,
                                                x = trainer.X_val,
                                                y = trainer.y_val)

custom_metrics

# COMMAND ----------

trainer.log_metrics(custom_metrics)

# COMMAND ----------

# MAGIC %md #### XGBoost training and logging

# COMMAND ----------

# MAGIC %md Confgure Tuner

# COMMAND ----------

hyperparameter_space = {'max_depth': hp.quniform('max_depth', 1, 10, 1),
                        'eval_metric': 'auc',
                        'early_stopping_rounds': 50}

tuner_args = {"hyperparameter_space": hyperparameter_space,
              "hyperopt_max_evals": 200 ,
              "hyperopt_iteration_stop_count": 20,
              "hyperopt_early_stopping_threshold": 0.05}

tuner = XGBoostHyperoptTuner(**tuner_args)

# COMMAND ----------

# MAGIC %md Tune and train

# COMMAND ----------

raw_features_table = 'default.mlc_raw_features'
label_col = 'Survived'

description = "An example Random Forest model trained with hyerpopt"

training_params = {"model": xgb.XGBClassifier,
                   "model_name": "random_forecast",
                   "model_type": "classifier",
                   "delta_feature_table": raw_features_table, 
                   "delta_train_val_id_table": f"{raw_features_table}_train", 
                   "train_size": 0.8, 
                   "preprocessing_pipeline": preprocessing_pipeline,
                   "label_col": label_col,
                   "tuner": tuner,
                   "mlflow_experiment_location": "/Shared/ml_production_experiment",
                   "mlflow_run_description": description}

trainer = SkLearnHyperoptTrainer(**training_params)
trainer.train()
