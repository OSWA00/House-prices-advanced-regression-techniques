"""
Th

Usage: 
    python3 ./scripts/etl.py

"""

import logging
from pathlib import Path
import logging


import click
import pandas as pd
import numpy as np


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def elt(config_file):
    
    # Configure logger
    logging.basicConfig(filename='./log/etl.log', level=logging.DEBUG)

    # Load config.yml
    logging.info("Config file: {}".format(config_file))



if __name__ == '__main__':
    elt()
