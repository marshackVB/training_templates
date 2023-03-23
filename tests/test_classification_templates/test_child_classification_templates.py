from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
from dev.trainer import SkLearnHyperoptBase
from dev.tests.sample_data_gen import sample_pandas_dataframe
#from dev.tests.conftest import sample_pandas_dataframe


class TestSklearnHyperoptBase(SkLearnHyperoptBase):
    """
    A subclass designed to test the functionality of SkLearnHyperoptBase
    """
    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.model = RandomForestClassifier

    
    def config_hyperopt_objective_fn(self, X_train_transformed, X_val_transformed):

        def hyperopt_objective_fn(params):
       
            params['n_estimators'] =   int(params['n_estimators'])

            model = self.init_model(params)

            model.fit(X_train_transformed, self.y_train.values.ravel())
        
            precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(self.y_val, 
                                                                                model.predict(X_val_transformed), 
                                                                                average='weighted')

            digits = 3
            metrics = OrderedDict()
            metrics["precision_val"]= round(precision_val, digits)
            metrics["recall_val"] =   round(recall_val, digits)
            metrics["f1_val"] =       round(f1_val, digits)

            return {'status': STATUS_OK, 'loss': 1- f1_val, 'metrics': metrics}
    
        return hyperopt_objective_fn
       

feature_df = sample_pandas_dataframe()

model_params = {"n_estimators":25, 
                "max_depth":5,
                "min_samples_split": 10}

training_args = {"feature_df": feature_df, 
                 "train_size": 0.8, 
                 "numerical_cols": ['Age', 'FareRounded'],
                 "categorical_cols": ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp'], 
                 "binary_cols": ['NameMultiple'], 
                 "label_col": 'Survived', 
                 "model_name": 'random_forest', 
                 "problem_type": 'classification', 
                 "hyperparameter_space": {'n_estimators': hp.quniform('n_estimators', 2, 10, 1)},
                 "max_evals": 5,
                 "experiment_location": "/Shared/ml_production_experiment", 
                 "description": "A Model instance designed for testing",
                 "commit_hash": None, 
                 "release_version": None, 
                 "random_state": 123, 
                 "shuffle": True}


def test_types_split_train_val_test():
  """
  Feature tables are Pandas DataFrames and label tables are
  Pandas Series
  """
  trainer = TestSklearnHyperoptBase(**training_args)
  trainer.train_test_split()
  
  assert isinstance(trainer.X_train, pd.core.frame.DataFrame)
  assert isinstance(trainer.X_val, pd.core.frame.DataFrame)
  
  assert isinstance(trainer.y_train, pd.core.frame.Series)
  assert isinstance(trainer.y_val, pd.core.frame.Series)


@pytest.mark.parametrize("train_size, random_state, shuffle", [(0.7, 123, False),
                                                               (0.6, 123, True),
                                                               (0.5, 999, True)])
def test_proportions_split_train_val_test(train_size, random_state, shuffle):
   
    updated_args = training_args
    updated_args["train_size"] = train_size
    updated_args["random_state"] = random_state
    updated_args["shuffle"] = shuffle

    trainer = TestSklearnHyperoptBase(**updated_args)
    trainer.train_test_split()

    # Train / val ratio is as expected
    X_train_cnt = trainer.X_train.shape[0]
    X_val_cnt = trainer.X_val.shape[0]
    y_train_cnt = trainer.y_train.shape[0]
    y_val_cnt = trainer.y_val.shape[0]
    combined_cnt = updated_args['feature_df'].shape[0]

    val_size = round(1 - train_size, 1)
    assert round(X_train_cnt / combined_cnt, 1) == train_size
    assert round(X_val_cnt / combined_cnt, 1) == val_size
    assert round(y_train_cnt / combined_cnt, 1) == train_size
    assert round(y_val_cnt / combined_cnt, 1) == val_size


def test_no_obs_overlap_split_train_val_test():

    trainer = TestSklearnHyperoptBase(**training_args)
    trainer.train_test_split()

    common_train_observations = set(trainer.X_train.index) & set(trainer.X_val.index)
    common_val_observations = set(trainer.y_train.index) & set(trainer.y_val.index)

    assert len(common_train_observations) == 0
    assert len(common_val_observations) == 0


def test_fit_predict_model_pipeline(training_args=training_args, model_params=model_params):
    """
    Model training pipeline component types are correct; all column types
    are passed to the model pipeline; model pipeline prediction works
    as expected.
    """
    trainer = TestSklearnHyperoptBase(**training_args)
    trainer.train_test_split()
    model_pipeline = trainer.init_model_pipeline(model_params)

    # Pipeline and component types are as expected
    assert isinstance(model_pipeline['preprocessing_pipeline'], ColumnTransformer)
    assert isinstance(model_pipeline['model'], RandomForestClassifier)
    assert isinstance(model_pipeline, Pipeline)

    # Numerical, categorical, and binary columns are all passed to ColumnTransformer
    column_transformations = {}
    for transformer in model_pipeline['preprocessing_pipeline'].transformers:
        column_transformations[transformer[0]] = transformer[2]

    for column_type, list_of_columns in column_transformations.items():
        assert(list_of_columns == training_args[column_type])

    model_pipeline.fit(trainer.X_train, trainer.y_train)
    predictions = model_pipeline.predict_proba(trainer.X_val)
    observations = trainer.X_val.shape[0]
    assert predictions.shape == (observations, 2)


def test_hyperopt_objective_fn(training_args=training_args, model_params=model_params):

    trainer = TestSklearnHyperoptBase(**training_args)
    trainer.train_test_split()

    X_train_transformed, X_val_transformed = trainer.transform_features_for_hyperopt()
    objective_fn = trainer.config_hyperopt_objective_fn(X_train_transformed, X_val_transformed)
    objective_results = objective_fn(model_params)

    assert objective_results['status'] == 'ok'
    assert type(objective_results['loss']) == np.float64
    assert isinstance(objective_results['metrics'], OrderedDict)


def test_hyperopt_search(training_args=training_args, model_params=model_params):

    trainer = TestSklearnHyperoptBase(**training_args)
    trainer.train_test_split()

    best_parameters = trainer.tune_hyperparameters()

    assert isinstance(best_parameters, dict)

def test_train():

    trainer = TestSklearnHyperoptBase(**training_args)
    trainer.train()

    logged_model = f"runs:/{trainer.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    predictions = loaded_model.predict(trainer.X_val)

    observations = trainer.X_val.shape[0]
    assert predictions.shape == (observations, )



    
    


