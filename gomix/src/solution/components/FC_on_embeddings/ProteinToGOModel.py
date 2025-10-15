import torch.nn as nn
import torch.nn.functional as F


class ProteinToGOModel(nn.Module):
    def __init__(self, protein_embedding_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(protein_embedding_size, 11000),
            nn.BatchNorm1d(11000),
            nn.ReLU(),

            nn.Linear(11000, 6000),
            nn.BatchNorm1d(6000),
            nn.ReLU(),

            nn.Linear(6000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),

            nn.Linear(4000, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        return F.sigmoid(self.forward(x))
