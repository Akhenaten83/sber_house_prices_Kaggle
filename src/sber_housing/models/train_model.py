
from typing import Any, Dict, Optional
import pandas as pd  
from sklearn.metrics import mean_squared_log_error,r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import optuna
from functools import partial

def train_catboost(X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None, optimize_hp: bool = False) -> CatBoostRegressor: 
    
    cat_features = list(X.select_dtypes('object').columns)
    
    if params is None:
        params = dict()
    if optimize_hp:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('starting hyper-parameters optimization')
        study = optuna.create_study(direction="minimize")
        study.optimize(partial(
            objective_catboost,
            cat_features=cat_features,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
            ),
            n_trials=100, n_jobs=4, timeout=3600)
        print("Best trial:")
        trial = study.best_trial
        print("RMSE: ", trial.value)
        print("Best hyperparameters: ", trial.params)
        params = trial.params

    regressor = CatBoostRegressor(cat_features=cat_features, **params)
    regressor.fit(X, y, verbose=100)
    return regressor

def objective_catboost(trial,cat_features,X_train,y_train,X_test,y_test):
    
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    depth = trial.suggest_int("depth", 4, 10)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 10, log=True)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    random_strength = trial.suggest_float("random_strength", 1e-3, 10, log=True)
    border_count = trial.suggest_int("border_count", 32, 255)

    model = CatBoostRegressor(
        cat_features=cat_features,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        random_strength=random_strength,
        border_count=border_count,
        loss_function="RMSE",
        iterations=1000,
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred = model.predict(X_test)
    msle = mean_squared_log_error(y_test, y_pred)

    return msle

