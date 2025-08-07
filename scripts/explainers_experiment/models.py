from abc import abstractmethod
from typing import Literal, Type
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import (
    GIN, GINConv, GINEConv, GATv2Conv,
    global_mean_pool, global_add_pool
)
from torch_geometric.data import Data


class Model(torch.nn.Module):
    def __init__(
        self,
        task_type: Literal['regression', 'classification'],
        **kwargs,
    ):
        super().__init__()
        if task_type not in ['regression', 'classification']:
            raise ValueError(f"Parameter task_type (given: {task_type}) must be either "
                             "'regression' or 'classification'.")
        self.task_type = task_type

    @abstractmethod
    def forward(self, x, edge_index, edge_attr, batch=None, return_representations=False):
        '''
        Returns:
            For classification, logits (for BCEWithLogitsLoss).
            For regression, raw values.
        '''
        pass

    def predict(self, data: Data, binarize: bool = True) -> torch.Tensor:
        raw_prediction = self(data.x, data.edge_index, data.edge_attr, data.batch)
        prediction = self.to_task_type_relevant_predictions(raw_prediction, binarize)
       #print(f"{raw_prediction=}")
        return prediction

    def to_task_type_relevant_predictions(self, logits: torch.Tensor, binarize: bool = True) -> torch.Tensor:
        '''
        For classification task type:
          * to compute a ranking metric (e.g., ROC AUC), set binarize = False;
          * to compute a classification accurracy metric, set binarize = True.
        '''
        match self.task_type:
            case 'classification':
                if binarize:
                    return torch.round(torch.sigmoid(logits))
                else:
                    return torch.sigmoid(logits)  # without torch.round()
            case 'regression' | _:
                return logits

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load(self, model_file_path: Path, device: torch.device):
        assert model_file_path.exists()
        self.load_state_dict(
             torch.load(model_file_path, weights_only=True),
             strict=True
        )
        if device:
            self.to(device)
        self.eval()
       #print(f"LOADED model from '{model_file_path}'.")


class ModelGIN(Model):
    """
    A class representing a Graph Isomorphism Network (GIN) model for molecular property prediction tasks.
    """
    def __init__(self, n_node_features: int, hidden_size: int, dropout: float, **kwargs):
        super().__init__(**kwargs)

        self.c1 = GIN(n_node_features, hidden_size, 1)
        self.l1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.c2 = GIN(hidden_size, hidden_size, 2)
        self.l2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.c3 = GIN(hidden_size, hidden_size, 2)
        self.l3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch=None, return_representations=False):
        out = self.c1(x, edge_index)
        out = self.l1(out)
        out = self.c2(out, edge_index)
        out = self.l2(out)
        out = self.c3(out, edge_index)
        out = self.l3(out)

        graph_representation = global_mean_pool(out, batch)
        if return_representations:
            return graph_representation

        final_output = self.out(graph_representation)
        final_output = final_output.flatten()

        return final_output


class ModelResidualGIN(Model):
    """
    GIN with residual connections to prevent over-smoothing.
    """
    def __init__(self, n_node_features: int, hidden_size: int, n_layers: int, dropout: float, **kwargs):
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.initial_proj = nn.Linear(n_node_features, hidden_size)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(), nn.Linear(hidden_size * 2, hidden_size))
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        # QUICKFIX for Grad-CAM exaplainer
        self.c3 = self.convs[-1]

        self.dropout = nn.Dropout(p=dropout)
        self.pool = global_add_pool
        self.out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x, edge_index, edge_attr, batch=None, return_representations=False):
        # project input features to the hidden dimension
        h = self.initial_proj(x)

        for i in range(self.n_layers):
            # store the input for the residual connection
            h_in = h

            # apply GIN block
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = nn.functional.relu(h)

            # add the residual connection
            h = h + h_in
            h = self.dropout(h)

        graph_representation = self.pool(h, batch)
        if return_representations:
            return graph_representation

        final_output = self.out(graph_representation)
        final_output = final_output.flatten()

        return final_output


class ModelEdgeGIN(Model):
    """
    GIN that uses GINEConv to incorporate bond/edge features.
    """
    def __init__(self, n_node_features: int, n_edge_features: int, hidden_size: int, dropout: float, **kwargs):
        super().__init__(**kwargs)

        nn1 = nn.Sequential(
            nn.Linear(n_node_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.c1 = GINEConv(nn1, edge_dim=n_edge_features)
        self.l1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.c2 = GINEConv(nn2, edge_dim=n_edge_features)
        self.l2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        nn3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.c3 = GINEConv(nn3, edge_dim=n_edge_features)
        self.l3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.pool = global_add_pool
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch=None, return_representations=False):
        out = self.c1(x, edge_index, edge_attr=edge_attr)
        out = self.l1(out)
        out = self.c2(out, edge_index, edge_attr=edge_attr)
        out = self.l2(out)
        out = self.c3(out, edge_index, edge_attr=edge_attr)
        out = self.l3(out)
        graph_representation = self.pool(out, batch)

        if return_representations:
            return graph_representation

        final_output = self.out(graph_representation)
        final_output = final_output.flatten()

        return final_output


class ModelGAT(Model):
    """
    Model using Graph Attention to weight neighbor importance.
    """
    def __init__(self, n_node_features: int, hidden_size: int, n_heads: int, dropout: float, **kwargs):
        super().__init__(**kwargs)

        self.c1 = GATv2Conv(n_node_features, hidden_size, heads=n_heads, dropout=dropout)
        self.l1 = nn.Sequential(nn.BatchNorm1d(hidden_size * n_heads), nn.LeakyReLU())

        self.c2 = GATv2Conv(hidden_size * n_heads, hidden_size, heads=n_heads, dropout=dropout)
        self.l2 = nn.Sequential(nn.BatchNorm1d(hidden_size * n_heads), nn.LeakyReLU())

        self.c3 = GATv2Conv(hidden_size * n_heads, hidden_size, heads=n_heads)
        self.l3 = nn.Sequential(nn.BatchNorm1d(hidden_size * n_heads), nn.LeakyReLU())

        self.pool = global_mean_pool
        self.out = nn.Sequential(
            nn.Linear(hidden_size * n_heads, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch=None, return_representations=False):
        out = self.c1(x, edge_index)
        out = self.l1(out)
        out = self.c2(out, edge_index)
        out = self.l2(out)
        out = self.c3(out, edge_index)
        out = self.l3(out)

        graph_representation = self.pool(out, batch)
        if return_representations:
            return graph_representation

        final_output = self.out(graph_representation)
        final_output = final_output.flatten()

        return final_output



MODEL_CLASS_REGISTRY = {
    'ModelGIN': ModelGIN,
    'ModelResidualGIN': ModelResidualGIN,
    'ModelEdgeGIN': ModelEdgeGIN,
    'ModelGAT': ModelGAT,
}


def get_model_class(mcls: str | Type[Model]) -> Type[Model]:
    if isinstance(mcls, str):
        cls = MODEL_CLASS_REGISTRY.get(mcls)
        if cls is None:
            raise ValueError(f"Unknown model class: '{mcls}'")
    elif issubclass(mcls, Model):
        cls = mcls
    else:
        raise TypeError('Argument model_class must be a string or a subclass of Model.')
    return cls


def load_model(
    model_class: str | Type[Model],
    task_type: Literal['regression', 'classification'],
    model_file_path: Path,
    device: torch.device,
    **model_kwargs,
) -> Model:
    """
    Load a model by class or name, with weights from file.

    Args:
        model_class: Either a string (e.g. 'ModelGIN') or a subclass of Model.
        model_file_path: Path to the saved model weights (a .pth file).
        model_kwargs: Keyword arguments needed to construct the model.

    Returns:
        An instance of the model with weights loaded.
    """
    model_class = get_model_class(model_class)
    model = model_class( task_type=task_type, **model_kwargs)
    model.load(model_file_path, device)
    assert model.num_trainable_parameters() > 0
    return model
