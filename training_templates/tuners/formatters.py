from typing import Callable, Dict, Union


def convert(hyperopt_params, convert_to_int_list):
    for param, value in hyperopt_params.items():
            if param in convert_to_int_list:
                hyperopt_params[param] = int(value)

    return hyperopt_params



def format_hyperopt_sklearn(hyperopt_params: Dict[str, Union[str, int, float]]) -> Dict[str, Union[str, int, float]]:
    """
    Reformat hyperopt parameters for sklearn
    """
    convert_to_int = [
    "n_estimators",
    "max_depth",
    "max_leaf_nodes"
    ]

    converted = convert(hyperopt_params, convert_to_int)
    return converted


def format_hyperopt_xbgoost(hyperopt_params: Dict[str, Union[str, int, float]])-> Dict[str, Union[str, int, float]]:
    """
    Reformat hyperopt parameters for xgboost
    """
    convert_to_int = [
    "n_estimators",
    "max_depth",
    "max_leaves",
    "max_bin",
    "grow_policy",
    ]
        
    converted = convert(hyperopt_params, convert_to_int)
    return converted


