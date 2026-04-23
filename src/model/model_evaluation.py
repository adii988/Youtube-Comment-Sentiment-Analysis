import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

from src.utils import Logger, load_params, load_data, load_model


logger = Logger("model_evaluation", logging.INFO)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return report, cm


def log_confusion_matrix(cm):

    os.makedirs(
        "reports",
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(
        "reports/figures/confusion_matrix_Test_Data.png"
    )

    plt.close()


def save_model_info(run_id):

    info = {
        "run_id": run_id,
        "model_path": "models/lgbm_model.joblib",
        "vectorizer_path": "tfidf_vectorizer.pkl"
    }

    with open(
        "experiment_info.json",
        "w"
    ) as file:
        json.dump(
            info,
            file,
            indent=4
        )


def main():

    try:
        logger.info(
            "Starting model evaluation..."
        )

        if not os.getenv(
            "GITHUB_ACTIONS"
        ):
            load_dotenv()

        dagshub_token = os.getenv(
            "DAGSHUB_PAT"
        )

        if dagshub_token:
            os.environ[
                "MLFLOW_TRACKING_USERNAME"
            ] = dagshub_token

            os.environ[
                "MLFLOW_TRACKING_PASSWORD"
            ] = dagshub_token

            mlflow.set_tracking_uri(
                "https://dagshub.com/dakshvandanarathi/YT-Sentiment-Analyser.mlflow"
            )

        mlflow.set_experiment(
            "dvc-pipeline-runs"
        )

        params = load_params(
            "params.yaml"
        )

        model = load_model(
            "models/lgbm_model.joblib"
        )

        vectorizer = load_model(
            "tfidf_vectorizer.pkl"
        )

        test_data = load_data(
            "data/interim/test_processed.csv"
        )

        with mlflow.start_run() as run:

            mlflow.log_params(params)
            
            test_data["comment"] = test_data["comment"].fillna("").astype(str)

            test_tfidf = vectorizer.transform(
                test_data["comment"]
            ).toarray()

            X_test = np.hstack([
                test_tfidf,
                test_data[
                    [
                        "word_count",
                        "char_count",
                        "avg_word_length"
                    ]
                ].values
            ])

            y_test = test_data[
                "category"
            ].values

            report, cm = evaluate_model(
                model,
                X_test,
                y_test
            )

            for label, metrics in report.items():

                if isinstance(
                    metrics,
                    dict
                ):

                    for metric_name, value in metrics.items():

                        mlflow.log_metric(
                            f"{label}_{metric_name}",
                            value
                        )

            log_confusion_matrix(cm)

            mlflow.log_artifact(
                "reports/confusion_matrix.png"
            )

            signature = infer_signature(
                X_test,
                model.predict(X_test)
            )

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature
            )

            save_model_info(
                run.info.run_id
            )

        logger.info(
            "Model evaluation completed."
        )

    except Exception as e:
        logger.critical(str(e))
        raise e


if __name__ == "__main__":
    main()