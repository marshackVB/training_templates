from typing import Any, Callable, Dict, Union, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_datasets(pandas_df: pd.DataFrame, label_col: str, train_size: float, 
                              shuffle: int, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a Pandas DataFrame of features into training and validation datasets
    """

    non_label_cols = [
        col for col in pandas_df.columns if col != label_col
    ]
    X_train, X_val, y_train, y_val = train_test_split(
        pandas_df[non_label_cols],
        pandas_df[label_col],
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return (X_train, X_val, y_train, y_val)