import pandas as pd
from src.utils.logging_config import logger  # Import the logger

try:
    logger.info("Starting data preprocessing...")

    df = pd.read_csv("data/raw/iris_raw.csv")
    logger.info("Successfully loaded raw data.")

    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    logger.info("Renamed columns.")

    df.to_csv("data/interim/iris_cleaned.csv", index=False)
    logger.info("Cleaned dataset saved.")

except Exception as e:
    logger.error(f"Error in preprocessing: {e}")
