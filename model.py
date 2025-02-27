import torch
import torch.nn as nn

class Transformer_Model(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        ff,
        n_layers,
        dropout,
        M,
        out_length,
        batch_first=True,
    ):
        super(Transformer_Model, self).__init__()
        self.model_type = "Transformer"

        self.gelu = nn.GELU()

        self.pos_embedding = nn.Embedding(M, d_model)
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
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers
        )

        self.projection_layer = nn.Sequential(
            nn.Linear(2, 8),
            self.gelu,
            nn.Linear(8, 16),
            self.gelu,
            nn.Linear(16, d_model),
            self.gelu,
        )

        self.out_layer = nn.Sequential(nn.Linear(d_model, 2))

        self.expand_layer = nn.Linear(M, out_length)

    def forward(self, src):

        src = self.projection_layer(src)

        src = src.permute(0, 2, 1)
        src = self.bn1(src)
        src = src.permute(0, 2, 1)

        B, T, n = src.shape

        idx = torch.arange(0, T).to(src.device)

        pe = self.pos_embedding(idx)

        src = src + pe

        src = src.permute(0, 2, 1)
        src = self.bn2(src)
        src = src.permute(0, 2, 1)

        logits = self.transformer_encoder(src)

        logits = logits.permute(0, 2, 1)

        logits = self.bn3(logits)

        logits = self.gelu(self.expand_layer(logits))

        logits = self.bn4(logits)

        logits = logits.permute(0, 2, 1)

        outputs = self.out_layer(logits)

        return outputs
