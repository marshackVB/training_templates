from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from training_templates.tuners  import XGBoostHyperoptTuner, SkLearnHyperoptTuner
from training_templates.data_utils import sample_pandas_dataframe, train_val_split


def transformed_features():
    """
    Split and transform sample Pandas DataFrame of features.
    """

    df = sample_pandas_dataframe()

    X_train, X_val, y_train, y_val = train_val_split(df, "Survived", 0.8)

    categorical_cols = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
    numerical_cols = ['Age', 'FareRounded']
    binary_cols = ['NameMultiple']

    numeric_transform = make_pipeline(SimpleImputer(strategy="most_frequent"))

    categorical_transform = make_pipeline(
        SimpleImputer(
            strategy="constant", fill_value="missing"
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

    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_val_transformed = preprocessing_pipeline.transform(X_val)

    return(X_train_transformed, X_val_transformed, y_train, y_val)


X_train_transformed, X_val_transformed, y_train, y_val = transformed_features()


def get_objective_fn_args(model):
    """
    Configure model and data attributes required for training
    """ 
    def init_model(model_params=None):
        if not model_params:
            return model()
        else:
            return model(**model_params)
        
    objective_args = {"init_model": init_model,
                      "X_train_transformed": X_train_transformed,
                      "X_val_transformed": X_val_transformed,
                      "y_train": y_train,
                      "y_val": y_val,
                      "random_state": 123}
    
    return objective_args
    

def get_hyperopt_tuner_args():
    """
    Configure comming hyperopt attributes
    """
    tuner_args = {"hyperopt_max_evals": 20,
                  "hyperopt_iteration_stop_count": 5,
                  "hyperopt_early_stopping_threshold": 0.05}
    return tuner_args


def test_sklearn_hyperopt_tuner():
    hyperparameter_space = {
        "n_estimators": hp.quniform("n_estimators", 2, 10, 1),
        "max_features": hp.uniform("max_features", 0.5, 1.0),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }
    hyperopt_tuner_args = get_hyperopt_tuner_args()
    hyperopt_tuner_args["hyperparameter_space"] = hyperparameter_space
    tuner  = SkLearnHyperoptTuner(**hyperopt_tuner_args) 
    objective_fn_args = get_objective_fn_args(RandomForestClassifier)
    best_params = tuner.tune(**objective_fn_args)

    assert isinstance(best_params, dict)
    assert type(best_params["n_estimators"]) == int
    assert type(best_params["max_features"]) == float
    assert type(best_params["criterion"]) == str


def test_xgboost_hyperopt_tuner():
    hyperparameter_space = {
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }
    hyperopt_tuner_args = get_hyperopt_tuner_args()
    hyperopt_tuner_args["hyperparameter_space"] = hyperparameter_space
    tuner = XGBoostHyperoptTuner(**hyperopt_tuner_args)
    objective_fn_args = get_objective_fn_args(xgb.XGBClassifier)
    best_params = tuner.tune(**objective_fn_args)

    assert isinstance(best_params, dict)
    assert type(best_params["max_depth"]) == int