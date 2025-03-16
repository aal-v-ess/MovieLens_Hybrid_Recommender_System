import logging
import yaml
# from dotenv import load_dotenv
import os

# load_dotenv()


def get_logger(name: str) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    return logger


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")

    if not config["HOPSWORKS_API_KEY"]:
        raise ValueError("API key not found! Set it in a .env file or environment variable.")


    return config