from hyperopt import hp
from hyperopt.pyll.base import scope
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from training_templates.tuners  import XGBoostHyperoptTuner, Tuner
from training_templates.data_utils import sample_pandas_dataframe, train_val_split


@pytest.fixture
def transformed_features(default_training_args):

    df = sample_pandas_dataframe()
    X_train, X_val, y_train, y_val = train_val_split(df, "Survived", 0.8)

    preprocessing_pipeline = default_training_args['preprocessing_pipeline']

    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_val_transformed = preprocessing_pipeline.transform(X_val)

    return(X_train_transformed, X_val_transformed, y_train, y_val)


def get_init_model_func(model):
    def init_model(model_params=None):
            if not model_params:
                return model()
            else:
                return model(**model_params)
            
    return init_model


@pytest.fixture
def objective_fn_args(transformed_features):

    X_train_transformed, X_val_transformed, y_train, y_val = transformed_features
 
    args = {"X_train_transformed": X_train_transformed,
            "X_val_transformed": X_val_transformed,
            "y_train": y_train,
            "y_val": y_val,
            "random_state": 123}
    
    return args


def test_sklearn_hyperopt_tuner(objective_fn_args, default_tuner):

    model_init = get_init_model_func(RandomForestClassifier)
    objective_fn_args['init_model'] = model_init

    best_params = default_tuner.tune(**objective_fn_args)

    assert isinstance(best_params, dict)
    assert type(best_params["n_estimators"]) == int
    assert type(best_params["max_features"]) == float
    assert type(best_params["criterion"]) == str


def test_xgboost_hyperopt_tuner(objective_fn_args, default_tuner_args):

    model = get_init_model_func(xgb.XGBClassifier)
    objective_fn_args['init_model'] = model

    hyperparameter_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }
    default_tuner_args["hyperparameter_space"] = hyperparameter_space

    model_name = "xgboost"
    tuner = Tuner.load_tuner(model_name, default_tuner_args)
    #tuner = XGBoostHyperoptTuner(**default_tuner_args)
    best_params = tuner.tune(**objective_fn_args)

    assert isinstance(best_params, dict)
    assert type(best_params["max_depth"]) == int