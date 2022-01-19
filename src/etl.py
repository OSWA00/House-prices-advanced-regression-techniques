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
from transform import transform_totalBsmtSF, log_transformation


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def elt(config_file):

    # Configure logger
    logging.basicConfig(filename='./log/etl.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config.yml
    logging.info("Config file: {}".format(config_file))
    config = parse_config(config_file)

    raw_data = config["etl"]["raw_data_file"]
    raw_data_no_labels = config["etl"]["raw_data_file_no_label"]

    logging.info("config: {}".format(config['etl']))

    # Data loading
    df_train = pd.read_csv(raw_data)
    df_test = pd.read_csv(raw_data_no_labels)

    target_variable = 'SalePrice'
    features = ['OverallQual', 'GrLivArea', 'GarageCars',
                'TotalBsmtSF', 'FullBath', 'YearBuilt']

    data_target_variable = df_train[target_variable]

    df_test = df_test[features]
    df_train = df_train[features]
    df_train[target_variable] = data_target_variable

    logging.info("Data loaded")

    # Deleting outliers
    df_train = df_train.drop(df_train[df_train['GrLivArea'] > 4000].index)
    df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 3000].index)

    logging.info("Removed outliers from trained data")

    # Normalize data
    features_to_transform = ['GrLivArea', 'SalePrice']
    df_train = log_transformation(df_train, features_to_transform)
    df_test = log_transformation(df_test, [features_to_transform[0]])

    df_train = transform_totalBsmtSF(df_train)
    df_test = transform_totalBsmtSF(df_test)

    logging.info("Normalized test & train data")

    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)

    logging.info("Convert categorical features to indicators")

    df_train.to_csv("./data/train.csv")
    df_test.to_csv("./data/test.csv")

    logging.info("train.csv  & test.csv updated")


if __name__ == '__main__':
    elt()
