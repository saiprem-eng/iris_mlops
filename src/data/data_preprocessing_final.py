from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import logging
from src.utils.logging_config import logger  

try:
    logger.info("Starting final data processing...")

    # Ensure the processed data directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Load cleaned data
    df = pd.read_csv("data/interim/iris_cleaned.csv")
    logger.info("Successfully loaded cleaned Iris dataset.")

    # Normalize features (all columns except the last one)
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    logger.info("Applied StandardScaler normalization to numerical features.")

    # Save final processed data
    df.to_csv("data/processed/iris_final.csv", index=False)
    logger.info("Final processed Iris dataset saved to 'data/processed/iris_final.csv'.")

except Exception as e:
    logger.error(f"Error in final processing: {e}")
