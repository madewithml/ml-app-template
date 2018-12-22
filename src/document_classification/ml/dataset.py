import os
import json
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader

from document_classification.configs.config import ml_logger
from document_classification.ml.vocabulary import Vocabulary, SequenceVocabulary
from document_classification.ml.vectorizer import Vectorizer

class Dataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer

        # Data splits
        self.train_df = self.df[self.df.split=='train']
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        self.lookup_dict = {'train': (self.train_df, self.train_size),
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

        # Class weights (for imbalances)
        class_counts = df.y.value_counts().to_dict()
        def sort_key(item):
            return self.vectorizer.y_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, df):
        train_df = df[df.split=='train']
        return cls(df, Vectorizer.from_dataframe(train_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return Vectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(
            self.target_split, self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        X = self.vectorizer.vectorize(row.X)
        y = self.vectorizer.y_vocab.lookup_token(
            row.y)
        return {'X': X, 'y': y}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self, batch_size, collate_fn, shuffle=True,
                         drop_last=False, device="cpu"):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                collate_fn=collate_fn, shuffle=shuffle,
                                drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

def sample(dataset):
    """Some sanity checks on the dataset.
    """
    sample_idx = random.randint(0,len(dataset))
    sample = dataset[sample_idx]
    ml_logger.info("\n==> 🔢 Dataset:")
    ml_logger.info("Random sample: {0}".format(sample))
    ml_logger.info("Unvectorized X: {0}".format(
        dataset.vectorizer.unvectorize(sample['X'])))
    ml_logger.info("Unvectorized y: {0}".format(
        dataset.vectorizer.y_vocab.lookup_index(sample['y'])))
