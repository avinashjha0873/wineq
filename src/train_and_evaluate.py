# -*- coding: utf-8 -*-
#load train and test
#train algo
#save metrics and parameters

import os
import pandas as pd
import numpy as np
import argparse
from get_data import read_params
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import joblib


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]
    fit_intercept = config["estimators"]["LinearRegression"]["params"]["fit_intercept"]
    normalize = config["estimators"]["LinearRegression"]["params"]["normalize"]
    models_dir = config["models_dir"]
    
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    
    train_X = train.drop(target, axis = 1)
    test_X = test.drop(target, axis = 1)
    
    train_y = train[target]
    test_y = test[target]
    
    model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
    model.fit(train_X, train_y)
    pred_train = model.predict(train_X)
    pred_test = model.predict(test_X)
    
    (mae, mse, r2) = evaluate(test_y, pred_test)
    

    ######################################################
    
    scores_file = config["reports"]["scores"]
    
    with open(scores_file, "w") as f:
        scores = {
            'mse': mse,
            "mae": mae,
            "r2": r2
            }
        
        json.dump(scores, f, indent = 4)
     
    params_file = config["reports"]["params"]
    with open(params_file, "w") as f:
        params = {
            'fit intercept': fit_intercept,
            'normalize': normalize
            }
        
        json.dump(params, f, indent=4)
     #################################################################   
     
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.joblib")
    
    joblib.dump(model, model_path)

      
        
def evaluate(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, mse, r2
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)