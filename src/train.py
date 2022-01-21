'''
Creates and trains model pipeline and exports to model path.

Usage:
    python3 ./src/train.py
    
'''

import logging
from pathlib import Path
from pickle import dump

import pandas as pd
import click

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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

    processed_data = Path(config["train"]['processed_train'])
    pipeline_params = config["train"]["pipeline_config"]
    model_path = Path(config["train"]["model_path"])

    logging.info("config: {}".format(config['train']))

    # Load proccessed data
    df_train = pd.read_csv(processed_data)

    X = df_train.drop(['SalePrice', 'Id'], axis=1)
    y = df_train['SalePrice']

    logging.info("Proccesed data loaded")

    # Create model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor())
    ])

    pipeline.set_params(**pipeline_params)

    logging.info("Pipeline created")
    logging.info("Pipeline paramaters {}".format(pipeline.get_params()))

    # Train model
    pipeline.fit(X, y)
    logging.info(f"Train score: {pipeline.score(X, y)}")
    logging.info(
        f"CV score: {cross_val_score(estimator = pipeline, X = X, y = y, cv = 5).mean()}"
    )
    # Export model
    with open(model_path, 'wb') as f:
        dump(pipeline, f)
    logging.info(f"Persisted model to {model_path}")


if __name__ == '__main__':
    train()
