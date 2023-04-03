
from typing import Any, Dict, Optional
import pandas as pd  
from sklearn.metrics import mean_squared_log_error,r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


def train_catboost(X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None) -> CatBoostRegressor: 
    if params is None:
        params = dict()

    regressor = CatBoostRegressor(cat_features=list(X.select_dtypes('object').columns), **params)
    regressor.fit(X, y, verbose=100)
    return regressor
