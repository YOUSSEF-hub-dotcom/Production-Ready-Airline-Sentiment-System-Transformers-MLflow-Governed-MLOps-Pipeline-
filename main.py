# main.py
import argparse
from data_pipeline import run_data_pipeline
from text_preprocessing import run_text_preprocessing
from eda_visualization import run_eda_visualization
from model import build_and_train_distilbert_model
from MLflow_lifeCycle import run_mlflow_full_lifecycle

import logging
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("       Airline Tweets Sentiment Analysis Project       ")
    logger.info("=" * 70)

    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\Tweets.csv"

    logger.info("\n>>> Step 1: Loading and Cleaning Raw Data")
    df = run_data_pipeline(FILE_PATH)

    logger.info("\n>>> Step 2: Text Cleaning and Preprocessing")
    df = run_text_preprocessing(df)

    logger.info("\n>>> Step 3: Exploratory Data Analysis & Visualization")
    df = run_eda_visualization(df)

    logger.info(f"\n>>> Step 4: Training DistilBERT (Epochs={args.epochs}, LR={args.lr})")
    model_results = build_and_train_distilbert_model(
        df,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger.info("\n>>> Step 5: Logging Model, Metrics, and Artifacts to MLflow")
    run_mlflow_full_lifecycle(model_results)

    logger.info("=" * 70)
    logger.info("      Project Completed Successfully! ✅       ")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()