import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path
import logging

from src.utils import Logger, load_params, load_data, save_data

logger = Logger("data_ingestion", logging.INFO)


# preprocess dataset
def preprocess_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    try:
        logger.info("Preprocessing data...")

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df = df[df["clean_comment"].str.strip() != ""]

        df.rename(columns={"clean_comment": "comment"}, inplace=True)

        if target_column in df.columns:
            df[target_column] = df[target_column].map({-1: 0, 0: 1, 1: 2})
        else:
            raise KeyError(f"{target_column} column not found")

        logger.info("Preprocessing completed")
        return df

    except Exception as e:
        logger.error(str(e))
        raise e


# split dataset
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info("Splitting dataset...")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        logger.info("Split completed")
        return train_df, test_df

    except Exception as e:
        logger.error(str(e))
        raise e


# main pipeline
def main():

    try:
        logger.info("Starting data ingestion pipeline...")

        # params.yaml read
        params = load_params("params.yaml")["data_ingestion"]

        # dataset load
        url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/main/data/reddit.csv"
        df = load_data(url)

        # preprocess
        final_df = preprocess_data(df, "category")

        # split
        train_df, test_df = split_data(
            final_df,
            "category",
            float(params["test_size"]),
            int(params["random_state"])
        )

        # save files
        save_data(train_df, "data/raw/train.csv")
        save_data(test_df, "data/raw/test.csv")

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.critical(str(e))
        raise e


if __name__ == "__main__":
    main()