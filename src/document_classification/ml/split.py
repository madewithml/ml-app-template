import os
import collections
import numpy as np
import pandas as pd

from document_classification.configs.config import ml_logger

def split_data(df, shuffle, train_size, val_size, test_size):
    """Split the data into train/val/test splits.
    """
    # Split by category
    by_category = collections.defaultdict(list)
    for _, row in df.iterrows():
        by_category[row.y].append(row.to_dict())
    ml_logger.info("\n==> 🛍️  Categories:")
    for category in by_category:
        ml_logger.info("{0}: {1}".format(category, len(by_category[category])))

    # Create split data
    final_list = []
    for _, item_list in sorted(by_category.items()):
        if shuffle:
            np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(train_size*n)
        n_val = int(val_size*n)
        n_test = int(test_size*n)

      # Give data point a split attribute
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:]:
            item['split'] = 'test'

        # Add to final list
        final_list.extend(item_list)

    # df with split datasets
    split_df = pd.DataFrame(final_list)
    ml_logger.info("\n==> 🖖 Splits:")
    ml_logger.info(split_df["split"].value_counts())

    return split_df