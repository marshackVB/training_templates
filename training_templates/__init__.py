__version__ = "0.0.1"

from training_templates.scikit_learn_templates.base_templates import SkLearnHyperoptBase
from training_templates.scikit_learn_templates.random_forest_hyperopt_template import (
    RandomForeastHyperoptTrainer,
)
from training_templates.scikit_learn_templates.xgboost_hyperopt_template import (
    XGBoostHyperoptTrainer,
)
