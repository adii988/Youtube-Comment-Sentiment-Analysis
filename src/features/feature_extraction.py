import numpy as np
import pandas as pd
import logging
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import Logger, load_params, load_data, save_data


logger = Logger("feature_extraction", logging.INFO)


def save_model(model, path):

    folder = os.path.dirname(path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(model, file)


def apply_tfidf(train_data, max_features, ngram_range):

    logger.info("TF-IDF transformation started...")

    train_data.dropna(inplace=True)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range)
    )

    X_train = vectorizer.fit_transform(
        train_data["comment"]
    ).toarray()

    extra_features = train_data[
        ["word_count", "char_count", "avg_word_length"]
    ].values

    final_train = np.hstack(
        [X_train, extra_features]
    )

    columns = list(
        vectorizer.get_feature_names_out()
    ) + [
        "word_count",
        "char_count",
        "avg_word_length"
    ]

    train_df = pd.DataFrame(
        final_train,
        columns=columns
    )

    # root folder madhe save
    save_model(
        vectorizer,
        "tfidf_vectorizer.pkl"
    )

    logger.info(
        "TF-IDF transformation completed."
    )

    return train_df


def main():

    try:
        logger.info(
            "Starting feature extraction pipeline..."
        )

        params = load_params(
            "params.yaml"
        )["feature_extraction"]

        train_data = load_data(
            "data/interim/train_processed.csv"
        )

        train_tfidf = apply_tfidf(
            train_data,
            params["max_features"],
            params["ngram_range"]
        )

        os.makedirs(
            "data/processed",
            exist_ok=True
        )

        save_data(
            train_tfidf,
            "data/processed/train_tfidf.csv"
        )

        target = train_data[
            ["category"]
        ].reset_index(drop=True)

        save_data(
            target,
            "data/processed/train_target.csv"
        )

        logger.info(
            "Feature extraction pipeline completed successfully."
        )

    except Exception as e:
        logger.critical(str(e))
        raise e


if __name__ == "__main__":
    main()