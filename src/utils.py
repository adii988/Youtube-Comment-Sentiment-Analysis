import logging
import yaml
import pandas as pd
import os
import pickle
import joblib


class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def debug(self, msg):
        self.logger.debug(msg)


def load_params(path="params.yaml"):
    with open(path, "r") as file:
        params = yaml.safe_load(file)
    return params


def load_data(path):
    return pd.read_csv(path)


def save_data(df, path):
    folder = os.path.dirname(path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)

    df.to_csv(path, index=False)


def save_yaml(data, path):
    folder = os.path.dirname(path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)

    with open(path, "w") as file:
        yaml.dump(data, file)


def read_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model(path):

    if path.endswith(".joblib"):
        return joblib.load(path)

    elif path.endswith(".pkl"):
        with open(path, "rb") as file:
            return pickle.load(file)

    else:
        raise ValueError("Unsupported file format")