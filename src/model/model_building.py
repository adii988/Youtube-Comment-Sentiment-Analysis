import logging
import lightgbm as lgb
import pandas as pd
import pickle
import os

from typing import Dict, Tuple

from src.utils import Logger, load_params, load_data


logger = Logger("model_building", logging.INFO)


def save_model(model, path):

    folder = os.path.dirname(path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(model, file)


def prepare_training_data(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:

    logger.info("Preparing training data...")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Mismatch between features and target rows"
        )

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: Dict
):

    logger.info("Training LightGBM model...")

    model = lgb.LGBMClassifier(
        **model_params
    )

    model.fit(X, y)

    logger.info(
        "Model trained successfully."
    )

    return model


def main():

    try:
        logger.info(
            "Starting model building pipeline..."
        )

        params = load_params(
            "params.yaml"
        )["model_building"]

        X = load_data(
            "data/processed/train_tfidf.csv"
        )

        y = load_data(
            "data/processed/train_target.csv"
        )["category"].astype(int)

        X, y = prepare_training_data(
            X, y
        )

        model = train_model(
            X, y, params
        )

        save_model(
            model,
            "lgbm_model.pkl"
        )

        logger.info(
            "Model building completed successfully."
        )

    except Exception as e:
        logger.critical(str(e))
        raise e


if __name__ == "__main__":
    main()