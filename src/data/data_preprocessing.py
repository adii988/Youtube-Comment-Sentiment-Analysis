import pandas as pd
import re
import nltk
import logging
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils import Logger, load_data, save_data


logger = Logger("data_preprocessing", logging.INFO)


def preprocess_and_create_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Preprocessing comments and creating features...")

        # remove null values
        df.dropna(inplace=True)

        # text clean
        df["comment"] = df["comment"].astype(str)
        df["comment"] = df["comment"].str.strip()
        df["comment"] = df["comment"].str.lower()

        # remove newline
        df["comment"] = df["comment"].apply(
            lambda x: re.sub(r"\n", " ", x)
        )

        # stopwords remove
        stop_words = set(stopwords.words("english")) - {
            "not", "no", "but", "however", "yet"
        }

        df["comment"] = df["comment"].apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in stop_words]
            )
        )

        # special chars remove
        df["comment"] = df["comment"].apply(
            lambda x: re.sub(r"[^A-Za-z0-9\s]", "", x)
        )

        # lemmatization
        lemmatizer = WordNetLemmatizer()

        df["comment"] = df["comment"].apply(
            lambda x: " ".join(
                [lemmatizer.lemmatize(word) for word in x.split()]
            )
        )

        # features
        df["word_count"] = df["comment"].apply(
            lambda x: len(x.split())
        )

        df["char_count"] = df["comment"].apply(len)

        df["avg_word_length"] = df["char_count"] / (
            df["word_count"] + 1
        )

        logger.info("Preprocessing completed successfully.")
        return df

    except Exception as e:
        logger.error(str(e))
        raise e


def main():

    try:
        logger.info("Starting data preprocessing pipeline...")

        nltk.download("stopwords")
        nltk.download("wordnet")

        # load data
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")

        # preprocess
        train_data = preprocess_and_create_features(train_data)
        test_data = preprocess_and_create_features(test_data)

        # create folder
        os.makedirs("data/interim", exist_ok=True)

        # save data
        save_data(train_data, "data/interim/train_processed.csv")
        save_data(test_data, "data/interim/test_processed.csv")

        logger.info("Data preprocessing pipeline completed.")

    except Exception as e:
        logger.critical(str(e))
        raise e


if __name__ == "__main__":
    main()