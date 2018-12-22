import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from document_classification.configs.config import ml_logger

class DocumentClassificationModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_input_channels,
                 num_channels, hidden_dim, num_classes, dropout_p,
                 padding_idx=0):
        super(DocumentClassificationModel, self).__init__()

        # Emebddings
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                       num_embeddings=num_embeddings,
                                       padding_idx=padding_idx)

        # Conv weights
        self.conv = nn.ModuleList([nn.Conv1d(num_input_channels, num_channels,
                                             kernel_size=f) for f in [2,3,4]])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_channels*3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, channel_first=False, apply_softmax=False):

        # Embed
        x_in = self.embeddings(x_in)

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        if not channel_first:
            x_in = x_in.transpose(1, 2)

        # Conv outputs
        z1 = self.conv[0](x_in)
        z1 = F.max_pool1d(z1, z1.size(2)).squeeze(2)
        z2 = self.conv[1](x_in)
        z2 = F.max_pool1d(z2, z2.size(2)).squeeze(2)
        z3 = self.conv[2](x_in)
        z3 = F.max_pool1d(z3, z3.size(2)).squeeze(2)

        # Concat conv outputs
        z = torch.cat([z1, z2, z3], 1)

        # FC layers
        z = self.dropout(z)
        z = self.fc1(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred


def initialize_model(config, vectorizer):
    """Initialize the model.
    """
    ml_logger.info("\n==> 🚀 Initializing model:")
    model = DocumentClassificationModel(
        embedding_dim=config["embedding_dim"],
        num_embeddings=len(vectorizer.X_vocab),
        num_input_channels=config["embedding_dim"],
        num_channels=config["cnn"]["num_filters"],
        hidden_dim=config["fc"]["hidden_dim"],
        num_classes=len(vectorizer.y_vocab),
        dropout_p=config["fc"]["dropout_p"],
        padding_idx=vectorizer.X_vocab.mask_index)
    ml_logger.info(model.named_modules)
    return model

