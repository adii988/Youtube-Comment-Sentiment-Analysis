import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(
            'Data loaded and NaNs filled from %s',
            file_path
        )
        return df

    except Exception as e:
        logger.error(
            'Error loading data from %s: %s',
            file_path, e
        )
        raise


def load_model(model_path: str):

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug(
            'Model loaded from %s',
            model_path
        )

        return model

    except Exception as e:
        logger.error(
            'Error loading model from %s: %s',
            model_path, e
        )
        raise


def load_vectorizer(
    vectorizer_path: str
) -> TfidfVectorizer:

    try:
        with open(
            vectorizer_path,
            'rb'
        ) as file:
            vectorizer = pickle.load(file)

        logger.debug(
            'TF-IDF vectorizer loaded from %s',
            vectorizer_path
        )

        return vectorizer

    except Exception as e:
        logger.error(
            'Error loading vectorizer from %s: %s',
            vectorizer_path, e
        )
        raise


def load_params(params_path: str) -> dict:

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug(
            'Parameters loaded from %s',
            params_path
        )

        return params

    except Exception as e:
        logger.error(
            'Error loading parameters from %s: %s',
            params_path, e
        )
        raise


def evaluate_model(
    model,
    X_test,
    y_test
):

    try:
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

        logger.debug(
            'Model evaluation completed'
        )

        return report, cm

    except Exception as e:
        logger.error(
            'Error during model evaluation: %s',
            e
        )
        raise


def log_confusion_matrix(
    cm,
    dataset_name
):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(
        f'Confusion Matrix for {dataset_name}'
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = (
        f'confusion_matrix_{dataset_name}.png'
    )

    plt.savefig(cm_file_path)

    mlflow.log_artifact(
        cm_file_path
    )

    plt.close()


def save_model_info(
    run_id: str,
    model_path: str,
    file_path: str
) -> None:

    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }

        with open(file_path, 'w') as file:
            json.dump(
                model_info,
                file,
                indent=4
            )

        logger.debug(
            'Model info saved to %s',
            file_path
        )

    except Exception as e:
        logger.error(
            'Error saving model info: %s',
            e
        )
        raise


def main():

    mlflow.set_tracking_uri(
        "http://13.51.166.199:5000"
    )

    mlflow.set_experiment(
        'dvc-pipeline-runs'
    )

    with mlflow.start_run() as run:

        try:
            root_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../'
                )
            )

            params = load_params(
                os.path.join(
                    root_dir,
                    'params.yaml'
                )
            )

            for key, value in params.items():
                mlflow.log_param(
                    key,
                    value
                )

            model = load_model(
                os.path.join(
                    root_dir,
                    'lgbm_model.pkl'
                )
            )

            vectorizer = load_vectorizer(
                os.path.join(
                    root_dir,
                    'tfidf_vectorizer.pkl'
                )
            )

            test_data = load_data(
                os.path.join(
                    root_dir,
                    'data/interim/test_processed.csv'
                )
            )

            # TFIDF Features
            X_test_tfidf = vectorizer.transform(
                test_data['comment'].values
            ).toarray()

            # Extra Numeric Features
            extra_features = test_data[
                [
                    'word_count',
                    'char_count',
                    'avg_word_length'
                ]
            ].values

            # Final Combined Features
            X_test = np.hstack(
                (
                    X_test_tfidf,
                    extra_features
                )
            )

            y_test = test_data[
                'category'
            ].values

            input_example = pd.DataFrame(
                X_test[:5]
            )

            signature = infer_signature(
                input_example,
                model.predict(X_test[:5])
            )

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )

            save_model_info(
                run.info.run_id,
                "lgbm_model",
                "experiment_info.json"
            )

            mlflow.log_artifact(
                os.path.join(
                    root_dir,
                    'tfidf_vectorizer.pkl'
                )
            )

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

                    mlflow.log_metrics({
                        f"test_{label}_precision":
                        metrics['precision'],

                        f"test_{label}_recall":
                        metrics['recall'],

                        f"test_{label}_f1-score":
                        metrics['f1-score']
                    })

            log_confusion_matrix(
                cm,
                "Test Data"
            )

            mlflow.set_tag(
                "model_type",
                "LightGBM"
            )

            mlflow.set_tag(
                "task",
                "Sentiment Analysis"
            )

            mlflow.set_tag(
                "dataset",
                "YouTube Comments"
            )

        except Exception as e:
            logger.error(
                f"Failed to complete model evaluation: {e}"
            )

            print(f"Error: {e}")


if __name__ == '__main__':
    main()