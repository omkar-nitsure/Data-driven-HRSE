import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda" if torch.cuda.is_available() else "cpu"


class Transformer_Model(nn.Module):
    def __init__(
        self,
        d_model,
        context_length,
        n_head,
        ff,
        n_layers,
        dropout,
        out_length,
        batch_first=True,
    ):
        super(Transformer_Model, self).__init__()
        self.model_type = "Transformer"

        self.pos_embedding = nn.Embedding(context_length, d_model)
        self.src_mask = None

        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.bn4 = nn.BatchNorm1d(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=batch_first,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers
        )

        self.projection_layer = nn.Sequential(nn.Linear(2, d_model))

        self.out_layer = nn.Sequential(
            nn.Linear(d_model, 2),
        )

        self.expand_layer = nn.Sequential(nn.Linear(context_length, out_length))

    def forward(self, src):

        src = F.gelu(self.projection_layer(src))
        src = src.permute(0, 2, 1)
        src = self.bn1(src)
        src = src.permute(0, 2, 1)

        B, T, n = src.shape

        idx = torch.arange(0, T).to(device)

        pe = self.pos_embedding(idx)

        src = src + pe

        src = src.permute(0, 2, 1)
        src = self.bn2(src)
        src = src.permute(0, 2, 1)

        logits = self.transformer_encoder(src)

        logits = logits.permute(0, 2, 1)

        logits = self.bn3(logits)

        logits = F.gelu(self.expand_layer(logits))

        logits = self.bn4(logits)

        logits = logits.permute(0, 2, 1)

        outputs = self.out_layer(logits)

        return outputs
