"""
Th

Usage: 
    python3 ./scripts/etl.py

"""

import logging
from pathlib import Path

import click
import pandas as pd
import numpy as np

from utility import parse_config

@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def elt(config_file):

    # Configure logger
    logging.basicConfig(filename='./log/etl.log', level=logging.DEBUG)

    # Load config.yml
    logging.info("Config file: {}".format(config_file))
    config = parse_config(config_file)

    raw_data = config["etl"]["raw_data_file"]
    raw_data_no_labels = config["etl"]["raw_data_file_no_label"]

    logging.info("config: {}".format(config['etl']))

    # Data transformation
    df_train = pd.read_csv(raw_data)
    df_test = pd.read_csv(raw_data_no_labels)

    target_variable = 'SalePrice'
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

    data_target_variable = df_train[target_variable]

    df_test = df_test[features]
    df_train = df_train[features]
    df_train[target_variable] = data_target_variable
    
    





if __name__ == '__main__':
    elt()
