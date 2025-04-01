import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from src.utils.logging_config import logger  # Import logger

def ensure_directory_exists(directory):
    """Ensure the directory exists."""
    os.makedirs(directory, exist_ok=True)

def train_model(df, test_size, n_estimators, random_state):
    """Train and save a machine learning model."""
    try:
        logger.info("Starting model training...")

        # Prepare data
        X = df.drop(columns=["species"])
        y = df["species"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        logger.info("Model trained successfully.")
        return model, X_test, y_test

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise e

if __name__ == "__main__":
    ensure_directory_exists("data/models")

    df = pd.read_csv("data/final/iris_final_features.csv")
    params = load_params('params.yaml')
    test_size = params['model_training']['test_size']
    n_estimators = params['model_training']['n_estimators']
    random_state = params['model_training']['random_state']
    model, X_test, y_test = train_model(df, test_size=test_size, n_estimators=n_estimators, random_state=random_state)

    # Save trained model using pickle
    with open("data/models/iris_model.pkl", "wb") as file:
        pickle.dump(model, file)
    logger.info("Model saved to data/models/iris_model.pkl")
