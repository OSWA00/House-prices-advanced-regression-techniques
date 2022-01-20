


import logging
from pathlib import Path

import pandas as pd
import click

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utility import parse_config

@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def train(config_file):
    # Configure logger
    logging.basicConfig(filename='./log/train.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config.yml
    logging.info("Config file: {}".format(config_file))
    config = parse_config(config_file)

    processed_data = config["train"]['processed_train']
    pipeline_params = config["train"]["pipeline_config"]
    model_path = config["train"]["model_path"]

    logging.info("config: {}".format(config['train']))


if __name__ == '__main__':
    train()