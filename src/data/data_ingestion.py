import pandas as pd
from sklearn import datasets
from src.utils.logging_config import logger  

try:
    logger.info("Starting Iris dataset loading process...")

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    logger.info("Successfully loaded Iris dataset.")

    df.to_csv("data/raw/iris_raw.csv", index=False)
    logger.info("Raw Iris dataset saved.")

except Exception as e:
    logger.error(f"Error in loading data: {e}")
