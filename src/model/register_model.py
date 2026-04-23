import json
import mlflow
import logging
from src.utils import Logger
import os
from dotenv import load_dotenv

# Initialize Logger
logger = Logger("register_model", logging.INFO)


def load_model_info(file_path: str) -> dict:

    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)

        logger.info(
            f"Model info loaded from {file_path}"
        )

        return model_info

    except Exception as e:
        logger.error(
            f"Error loading model info: {e}"
        )
        raise


def register_model(model_name: str, model_info: dict) -> None:

    try:
        # Only model register
        model_uri = (
            f"runs:/{model_info['run_id']}/"
            f"{model_info['model_path']}"
        )

        model_version = mlflow.register_model(
            model_uri,
            model_name + "_lgbm_model"
        )

        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name + "_lgbm_model",
            version=model_version.version,
            stage="Staging"
        )

        logger.info(
            f"Model registered version "
            f"{model_version.version}"
        )

    except Exception as e:
        logger.error(
            f"Error during registration: {e}"
        )
        raise


def main():

    try:
        if not os.getenv(
            "GITHUB_ACTIONS"
        ):
            load_dotenv()

        dagshub_token = os.getenv(
            "DAGSHUB_PAT"
        )

        mlflow.set_tracking_uri(
            "http://13.51.166.199:5000"
        )

        if dagshub_token:
            os.environ[
                "MLFLOW_TRACKING_USERNAME"
            ] = dagshub_token

            os.environ[
                "MLFLOW_TRACKING_PASSWORD"
            ] = dagshub_token

        model_info = load_model_info(
            "experiment_info.json"
        )

        register_model(
            "sentiment_analysis",
            model_info
        )

        logger.info(
            "Model registration completed."
        )

    except Exception as e:
        logger.error(
            f"Failed registration process: {e}"
        )
        print(f"Error: {e}")


if __name__ == "__main__":
    main()