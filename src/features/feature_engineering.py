import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from src.utils.logging_config import logger  # Import logger

def ensure_directory_exists(directory):
    """Ensure the directory exists."""
    os.makedirs(directory, exist_ok=True)

def feature_engineering(df, scale_mean, scale_std):
    """Apply feature engineering steps."""
    try:
        logger.info("Starting feature engineering...")

        # Create new feature: petal area
        df["petal_area"] = df["petal_length"] * df["petal_width"]
        logger.info("Created new feature: petal_area.")

        # Scale numerical features
        scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
        df.iloc[:, :-2] = scaler.fit_transform(df.iloc[:, :-2])  # Exclude last 2 columns (species & petal_area)
        logger.info("Feature scaling applied.")

        return df

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise e

if __name__ == "__main__":
    ensure_directory_exists("data/final")

    df = pd.read_csv("data/processed/iris_final.csv")
    params = load_params('params.yaml')
    scale_mean = params['feature_engineering']['scale_mean']
    scale_std = params['feature_engineering']['scale_std']

    df = feature_engineering(df, scale_mean=scale_mean, scale_std=scale_std)
    df.to_csv("data/final/iris_final_features.csv", index=False)
    logger.info("Final dataset with features saved to data/final/iris_final_features.csv")
    logger.info("Feature engineering completed successfully.")