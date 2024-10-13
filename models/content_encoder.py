# models/content_encoder.py
import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(self, input_dim=1024, content_dim=256):
        super(ContentEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, content_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)
