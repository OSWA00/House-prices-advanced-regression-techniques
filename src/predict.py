'''
Predicts and exports from processed data

Usage:
    python3 ./src/predict.py
'''
import logging
from pathlib import Path
from pickle import load

import pandas as pd
import click
import numpy as np

from utility import parse_config


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def predict(config_file):
    # Configure logger
    logging.basicConfig(filename='./log/predict.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config.yml
    logging.info("Config file: {}".format(config_file))
    config = parse_config(config_file)

    processed_data = Path(config["predict"]['processed_test'])
    model_path = Path(config["predict"]["model_path"])
    export_path = Path(config["predict"]["export_path"])

    logging.info("config: {}".format(config['predict']))

    # Load data 
    X = pd.read_csv(processed_data)
    y_predicted = pd.DataFrame(columns=['SalePrice', 'Id'])
    y_predicted['Id'] = X['Id']
    X = X.drop('Id', axis=1)

    logging.info("Data loaded")

    # Load model
    f = open(model_path, "rb")
    trained_model = load(f)
    f.close()

    logging.info("Trained model loaded from {}".format(model_path))

    # Predict
    y_predicted['SalePrice'] = trained_model.predict(X)
    y_predicted['SalePrice'] = np.exp(y_predicted['SalePrice'])
    logging.info("Prediction done")

    # Export result
    y_predicted.to_csv(export_path, index=False)
    logging.info("Output file written to {}".format(export_path))


if __name__ == "__main__":
    predict()
