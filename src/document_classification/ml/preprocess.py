import os
import re

from document_classification.configs.config import ml_logger

def preprocess_text(text):
    """Basic text preprocessing.
    """
    text = ' '.join(word.lower() for word in text.split(" "))
    text = text.replace('\n', ' ')
    text = re.sub(r"[^a-zA-Z.!?_]+", r" ", text)
    text = text.strip()
    return text


def preprocess_data(df):
    df.X = df.X.apply(preprocess_text)
    ml_logger.info("\n==> 🚿 Preprocessing:")
    ml_logger.info(df.head())
    return df

