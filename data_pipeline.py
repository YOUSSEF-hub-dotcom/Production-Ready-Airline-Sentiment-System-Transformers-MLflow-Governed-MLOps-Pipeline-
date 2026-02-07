# data_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def load_data(file_path):
    logger.info("Loading Data .....")
    df = pd.read_csv(file_path)
    return df


def basic_data_overview(df):
    pd.set_option('display.width', None)

    print(df.head(11))

    logger.info('=======================>>> Analysis Function')
    logger.info("Information about Data:")
    print(df.info())
    logger.info("Statistical Data Analysis:")
    print(df.describe().round(2))
    logger.info('Number of rows and columns:')
    print(df.shape)
    logger.info('Columns in Data:')
    print(df.columns)
    logger.info("Frequency of Rows in Data (Duplicates):")
    print(df.duplicated().sum())

def clean_data(df):
    logger.info("=======================>>> Cleaning Data")

    logger.info("Remove Duplicates")
    df = df.drop_duplicates(keep='first')
    logger.info(f"Shape after removing duplicates: {df.shape}")

    logger.info("Missing Values Before Cleaning:")
    print(df.isnull().sum())

    missing_values = df.isnull().mean() * 100
    logger.info(f"The Percentage of missing values in data:\n {missing_values.round(2)}")

    logger.info(
        "The percentage of missing Value in negativereason column is: ~37% \n",
        "The percentage of missing Value in negativereason_confidence column is: ~28% \n",
        "The percentage of missing Value in tweet_location column is: ~32% \n",
        "The percentage of missing Value in user_timezone column is: ~32% \n",
        "We will fill these with Mode/Median"
    )

    df['negativereason'] = df['negativereason'].fillna(df['negativereason'].mode()[0])
    df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].median())
    df['tweet_location'] = df['tweet_location'].fillna(df['tweet_location'].mode()[0])
    df['user_timezone'] = df['user_timezone'].fillna(df['user_timezone'].mode()[0])

    logger.info("Missing values filled for key columns")


    logger.info(
        "The Percentage of missing value in tweet_coord column is: ~93% \n",
        "The Percentage of missing value in negativereason_gold column is: ~99% \n",
        "The Percentage of missing value in airline_sentiment_gold column is: ~99% \n",
        "We will DROP these columns"
    )

    df = df.drop(columns=['tweet_coord', 'negativereason_gold', 'airline_sentiment_gold'], axis=1, errors='ignore')

    logger.info("Columns dropped successfully")

    logger.info("Missing Values After Full Cleaning:")
    print(df.isnull().sum())

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values in Tweets Dataset After Cleaning')
    plt.tight_layout()
    plt.show()

    return df


def run_data_pipeline(file_path):
    logger.info("============ Starting Sentiment Analysis Data Pipeline ============")

    df = load_data(file_path)

    basic_data_overview(df)

    df_cleaned = clean_data(df)

    logger.info(f"Final Dataset Shape: {df_cleaned.shape}")
    logger.info("Final Columns:")
    print(df_cleaned.columns.tolist())

    logger.info("============ Data Pipeline Completed Successfully! ============")

    return df
