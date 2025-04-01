import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from src.utils.logging_config import logger  # Import logger

def load_model(model_path):
    """Load the trained model from pickle file."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    try:
        logger.info("Starting model evaluation...")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)

        return accuracy, report

    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise e

if __name__ == "__main__":
    df = pd.read_csv("data/final/iris_final_features.csv")
    X_test = df.drop(columns=["species"])
    y_test = df["species"]

    model = load_model("data/models/iris_model.pkl")
    evaluate_model(model, X_test, y_test)
    logger.info("Model evaluation completed.")                  