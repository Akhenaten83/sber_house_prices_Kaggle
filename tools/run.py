from sber_housing import build_features
from sber_housing import train
from sber_housing import model_eval
import pandas as pd
from pathlib import Path
from typing import Union,Optional
from datetime import datetime
from sber_housing.constants import DATETIME_FORMAT,TARGET_COL
import argparse
import json
import joblib

def main(
    train_data: pd.DataFrame,
    features_out: Union[str,Path],
    test_data: Optional[Union[str,Path]] = None,
    model_out: Optional[Union[str,Path]] = None,
    optimize_hp: bool = False,
    submission_out:Optional[Union[str,Path]] = None
    ) -> None:
    """
    
    """
    current_time = datetime.now().strftime(DATETIME_FORMAT)
    train_X,train_y = build_features(train_data, return_y=True)
    pd.concat((train_X,train_y),axis=1).to_csv(Path(features_out,f'{current_time}_train.csv'))
    
    if model_out is not None:

        metrics = model_eval(train_X,train_y)
        with open (Path(model_out,f'{current_time}_metrics.json'),'w') as  file:
            json.dump(metrics,file)

        model = train(train_X,train_y,optimize_hp=optimize_hp)
        with open(Path(model_out,f'{current_time}_model.joblib'),'wb') as file:
            joblib.dump(model,file)

    if test_data is not None:
        test_X = build_features(test_data)
        test_X['product_type']=test_X['product_type'].fillna('Investment')
        test_X.to_csv(Path(features_out,f'{current_time}_test.csv'))
        
        if submission_out is not None:
            y_pred = pd.Series(model.predict(test_X),index=test_X.index,name = 'price_doc')
            y_pred = y_pred.to_frame()
            y_pred.to_csv(Path(submission_out,f'{current_time}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('-f','--features_out', required=True)
    parser.add_argument('--test_data_path', required=False, default=None)
    parser.add_argument('-m','--model_out',required=False, default=None)
    parser.add_argument('-o','--optimize_hp',action ='store_true')
    parser.add_argument('-s','--submission_out',required=False,default=None)

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_data_path)
    if args.test_data_path is not None:
        test_data = pd.read_csv(args.test_data_path)
    else:
        test_data = None
    
    main(
        train_data=train_data,
        features_out=args.features_out,
        test_data=test_data,
        model_out=args.model_out,
        submission_out=args.submission_out,
        optimize_hp=args.optimize_hp
    )
