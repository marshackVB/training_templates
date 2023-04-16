from typing import Callable, Dict, Union

def format_hyperopt_sklearn(hyperopt_params: Dict[str, Union[str, int, float]], model: Callable) -> Dict[str, Union[str, int, float]]:
    """
    Reformat hyperopt parameters for sklearn
    """

    for param, value in hyperopt_params.items():
        defaul_value = model.__dict__[param]
        if isinstance(defaul_value, int):
            hyperopt_params[param] = int(hyperopt_params[param])
    return hyperopt_params


def format_hyperopt_xbgoost(hyperopt_params: Dict[str, Union[str, int, float]])-> Dict[str, Union[str, int, float]]:
    """
    Reformat hyperopt parameters for sklearn
    """
        
    convert_to_int = [
    "n_estimators",
    "max_depth",
    "max_leaves",
    "max_bin",
    "grow_policy",
]
        
    for param, value in hyperopt_params.items():
        if param in convert_to_int:
            hyperopt_params[param] = int(value)

    return hyperopt_params


