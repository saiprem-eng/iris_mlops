import logging
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/iris.log",  # Log file path
    filemode="a",  # Append logs instead of overwriting
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Set log level (INFO, DEBUG, ERROR, etc.)
)

# Create and export a logger instance
logger = logging.getLogger("iris_logger")  # Global logger