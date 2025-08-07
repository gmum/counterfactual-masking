import os

import torch
import torch.nn as nn
from torch_geometric.nn import GIN, GCNConv, global_mean_pool


class GraphIsomorphismNetwork(torch.nn.Module):
    """
    A class representing a Graph Isomorphism Network (GIN) model for molecular property prediction tasks.
    """

    def __init__(
            self,
            n_input_features: int,
            hidden_size: int,
            dropout: float = 0.3,
            path: str = "./gin"
    ):
        super().__init__()
        self.n_input_features = n_input_features
        self.path = path
        self.c1 = GIN(self.n_input_features, hidden_size, 1)
        self.l1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(p=dropout)
        )
        self.c2 = GIN(hidden_size, hidden_size, 2)
        self.l2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(p=dropout)
        )
        self.c3 = GIN(hidden_size, hidden_size, 2)
        self.l3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, edge_index, batch, return_representations=False):
        out = self.c1(x, edge_index)
        out = self.l1(out)
        out = self.c2(out, edge_index)
        out = self.l2(out)
        out = self.c3(out, edge_index)
        out = self.l3(out)
        out = global_mean_pool(out, batch)
        if return_representations:
            return out
        out = self.out(out)
        out = out.flatten()
        return out

    def predict(self, model, data, return_representations=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return model(x, edge_index, batch, return_representations)

    def load_model(self, name="best_model"):
        self.load_state_dict(torch.load(os.path.join(self.path, f"{name}.pth")), strict=False)

    def save_model(self, name="best_model"):
        """
        Saves the trained model's state to a file.
        """
        torch.save(self.state_dict(), os.path.join(self.path, f"{name}.pth"))

class GraphIsomorphismNetwork_classification(torch.nn.Module):
    """
    A class representing a Graph Isomorphism Network (GIN) model for molecular property prediction tasks.
    """

    def __init__(
            self,
            n_input_features: int,
            hidden_size: int,
            dropout: float = 0.3,
            path: str = "./gin"
    ):
        super().__init__()
        self.n_input_features = n_input_features
        self.path = path
        self.c1 = GIN(self.n_input_features, hidden_size, 1)
        self.l1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(p=dropout)
        )
        self.c2 = GIN(hidden_size, hidden_size, 2)
        self.l2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(p=dropout)
        )
        self.c3 = GIN(hidden_size, hidden_size, 2)
        self.l3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch, return_representations=False):
        out = self.c1(x, edge_index)
        out = self.l1(out)
        out = self.c2(out, edge_index)
        out = self.l2(out)
        out = self.c3(out, edge_index)
        out = self.l3(out)
        out = global_mean_pool(out, batch)
        if return_representations:
            return out
        out = self.out(out)
        out = out.flatten()
        return out

    def predict(self, model, data, return_representations=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return model(x, edge_index, batch, return_representations)

    def load_model(self, name="best_model"):
        self.load_state_dict(torch.load(os.path.join(self.path, f"{name}.pth")), strict=False)

    def save_model(self, name="best_model"):
        """
        Saves the trained model's state to a file.
        """
        torch.save(self.state_dict(), os.path.join(self.path, f"{name}.pth"))