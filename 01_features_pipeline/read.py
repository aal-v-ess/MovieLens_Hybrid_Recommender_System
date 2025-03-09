import pandas as pd
from pandas.errors import EmptyDataError
from typing import List
import os

import utils

logger = utils.get_logger(__name__)



def read_data(path: str,  file_name: str, drop_columns: List[str] = None) -> pd.DataFrame:
    """
    Read the data from the CSV file
    """
    file_path = path + file_name
    if not os.path.exists(file_path):
        logger.info(f"File not found: {file_path}")
    else:
        logger.info(f"Reading data from : {file_path}")
        try:
            if drop_columns:
                data = pd.read_csv(file_path, delimiter=",", low_memory=False).drop(columns=drop_columns)
            else:
                data = pd.read_csv(file_path, delimiter=",", low_memory=False)
        except EmptyDataError:
            file_path.unlink(missing_ok=True)
            
            raise ValueError(f"Downloaded file at {file_path} is empty. Could not load it into a DataFrame.")

    return data