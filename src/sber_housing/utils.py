from sklearn.model_selection import train_test_split
from sber_housing import train
import pandas as pd
from typing import Dict
from sklearn.metrics import r2_score,mean_squared_log_error


def model_eval(X: pd.DataFrame, y: pd.Series, random_state: int = 42, test_size: float = 0.15) -> Dict[str,float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    regressor = train(X_train,y_train)
    y_pred_test = regressor.predict(X_test) 
    y_pred_train = regressor.predict(X_train)

   
    rmsle_test = mean_squared_log_error(y_test, y_pred_test, squared=False)
    rmsle_train  = mean_squared_log_error(y_train,y_pred_train,squared=False)

    r2_test = r2_score(y_test,y_pred_test)
    r2_train = r2_score(y_train,y_pred_train)

    result = {
        'rmsle_test':rmsle_test,
        'rmsle_train': rmsle_train,
        'r2_test':r2_test,
        'r2_train':r2_train
        }
    
    return result
