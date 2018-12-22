import os
import pandas as pd

from document_classification.configs.config import ml_logger

def load_data(data_file):
    """Load data from CSV to Pandas DataFrame.
    """
    # Load into DataFrame
    df = pd.read_csv(data_file, header=0)
    ml_logger.info("\n==> 🍣 Raw data:")
    ml_logger.info(df.head())

    return df