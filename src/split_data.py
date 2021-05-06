# -*- coding: utf-8 -*-

#split raw data
# save it in data/processed

import os
import argparse
import pandas as pd
from get_data import read_params
from sklearn.model_selection import train_test_split

def split_save_data(config_path):
    config = read_params(config_path)
    raw_data_path = config["load_data"]["raw_data"]
    test_data_path = config["split_data"]["train_path"]
    train_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    test_ratio = config["split_data"]["test_size"]
    df = pd.read_csv(raw_data_path)
    train, test = train_test_split(
                                    df,
                                    random_state = random_state, 
                                    test_size=test_ratio)
    train.to_csv(train_data_path, index = False, encoding = "utf-8")
    test.to_csv(test_data_path, index = False, encoding = "utf-8")
    
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_save_data(config_path=  parsed_args.config)
