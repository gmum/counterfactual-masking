from typing import Literal
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ExplanationType, ModelMode, MaskType
from captum.attr import IntegratedGradients
from models import Model, ModelEdgeGIN

'''
Node importance functions

The functions return directional importances by default.
'''


def saliency_map(
    model: Model,
    data: Data,
    only_magnitude: bool = False
) -> torch.Tensor:
    '''
    Computes a saliency map for a graph using gradients with respect to node features.

    Args:
        model (Model): A trained GNN model.
        data (Data): Input graph data (with `x`, `edge_index`, `edge_attr`, `batch`).
        only_magnitude (bool): If True, return only L2 magnitude of gradients
                               (disregards direction).

    Returns:
        Tensor: A 1-dim tensor of size [n_nodes], with importance scores per node.
    '''
    # set model to inference mode (turn off dropout and batch normalization)
    model.eval()

    # ensure the autograd engine tracks gradients on node features
    data.x.requires_grad_()

    # zero any pre-existing input data node gradients
    if data.x.grad is not None:
        data.x.grad.zero_()

    # forward pass => prediction
    output = model(data.x, data.edge_index, data.edge_attr, data.batch)

    # should you consider output.sum().backward() for batched data?
    assert output.numel() == 1, 'Model output is not a single scalar (graph-level prediction).'

    # backward pass => gradient of output computed w.r.t. all gradient-tracking tensors
    output.backward()

    # retrieve gradients, shape: [n_nodes, n_node_features]
    grad = data.x.grad
    assert grad is not None, 'Gradients w.r.t. inpute were not computed.'

    # aggregate saliency per node
    if only_magnitude:
        # L2 norm across feature dimensions per node
        saliency = grad.norm(p=2, dim=1)
    else:
        # positive and negative gradient contributions
        pos_grad = F.relu( grad)
        neg_grad = F.relu(-grad)

        # difference between the norm of the positive gradients and the norm of the negative gradients
        saliency = pos_grad.norm(p=2, dim=1) - neg_grad.norm(p=2, dim=1)

    # stop tracking gradients for input
    data.x.requires_grad_(False)

    return saliency


def integrated_gradients(
    model: Model,
    data: Data,
    only_magnitude: bool = False,
    n_steps: int = 100,
) -> torch.Tensor:
    '''
    Computes directional node-level importance scores using Integrated Gradients.

    Args:
        model (Model): GNN model with graph-level output.
        data (Data): PyG Data object.
        only_magnitude (bool): If True, return unsigned scores.
        n_steps (int): Number of interpolation steps.

    Returns:
        Tensor: A 1-dim tensor of size [n_nodes], with importance scores per node.
    '''
    # set model to inference mode (turn off dropout and batch normalization)
    model.eval()

    # Captum will take care of attaching gradients internally
    x = data.x.detach()

    # baseline
    baseline_x = torch.zeros_like(x)

    # forward wrapper
    def _model_forward(x_input):
        return model(
            x_input,
            data.edge_index,
            data.edge_attr.detach() if isinstance(model, ModelEdgeGIN) else None,
            data.batch,
        )

    # node attributions
    ig = IntegratedGradients(_model_forward)
    attributions = ig.attribute(inputs=x, baselines=baseline_x, n_steps=n_steps)

    # attribution aggregation
    if only_magnitude:
        node_scores = attributions.norm(p=2, dim=1)
    else:
        pos_grad = F.relu( attributions)
        neg_grad = F.relu(-attributions)
        node_scores = pos_grad.norm(p=2, dim=1) - neg_grad.norm(p=2, dim=1)

    return node_scores


class _GradCAM:
    '''
    Grad-CAM for Graph Neural Networks.

    Captures node-level importance scores based on activations and gradients
    in a specified (intermediate) GNN layer (e.g., GCNConv or GINConv).
    '''
    def __init__(self, model: Model, target_layer_name: str = 'c3'):
        '''
        Initializes the Grad-CAM explainer for a given model.

        Args:
            model (Model): The GNN model to explain.
            target_layer_name (str): Name of the target convolutional layer
                                     whose activations and gradients will be
                                     used for computing Grad-CAM.
        '''
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None  # shape: [n_nodes, hidden_size]
        self.activations = None  # shape: [n_nodes, hidden_size]
        target_layer = getattr(self.model, target_layer_name)

        # During the forward pass, intercept the output of the target_layer.
        target_layer.register_forward_hook(
            lambda _m, _i, o: setattr(self, 'activations', o.detach())
        )

        # During the backward pass, intercept the gradient flowing back into the target_layer.
        target_layer.register_full_backward_hook(
            lambda _m, _gi, go: setattr(self, 'gradients', go[0].detach())
        )

    def node_importances(self, data: Data, only_magnitude: bool = False) -> torch.Tensor:
        '''
        Computes node-level importance scores using Grad-CAM.

        Args:
            data (Data): A PyG graph object containing the input to the model.
            only_magnitude (bool): If True, returns the absolute value of scores
                                   (disregards direction).

        Returns:
            Tensor: A 1D tensor of size [n_nodes], with importance scores per node.
        '''
        # track model gradients
        for param in self.model.parameters():
            param.requires_grad_(True)

        # set model to inference mode (turn off dropout and batch normalization)
        self.model.eval()

        # forward pass => prediction (hook!)
        output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)

        # should you consider output.sum().backward() for batched data?
        assert output.numel() == 1, 'Model output is not a single scalar (graph-level prediction).'

        # zero any pre-existing model gradients
        self.model.zero_grad()

        # backward pass => gradient of output computed w.r.t. all gradient-tracking tensors (hook)
        output.backward()

        # compute the average gradient for each feature channel across all nodes
        feature_weights = torch.mean(self.gradients, dim=0)  # shape: [hidden_size]

        # weighted sum of the activation maps
        cam = (self.activations * feature_weights).sum(dim=1)  # shape: [n_nodes]

        # stop tracking model gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        # do not normalize before returning
        return cam.abs() if only_magnitude else cam


def gradcam_node_importances(model: Model, data: Data, only_magnitude: bool = False) -> torch.Tensor:
    '''
    Convenience function to compute Grad-CAM node importances for a model.

    Args:
        model (Model): The GNN model to explain.
        data (Data): Input graph data.
        only_magnitude (bool): Whether to return unsigned (absolute) importances.

    Returns:
        Tensor: Importance scores per node in the graph.
    '''
    gradcam = _GradCAM(model)
    node_importances = gradcam.node_importances(data, only_magnitude)
    return node_importances


def gnnexplainer_node_importances(
    model: Model,
    data: Data,
    only_magnitude: bool = False,
    node_masking: Literal['object', 'attributes'] = 'object',
    sign_source: Literal['saliency', 'gradcam'] = 'gradcam',
    n_epochs: int = 100,
) -> torch.Tensor:
    '''
    Computes node importances using GNNExplainer with optional directional information.

    Args:
        model (Model): Trained GNN model.
        data (Data): A PyG graph input (must be graph-level task).
        only_magnitude (bool): If True, returns unsigned importances.
        sign_source (str): If not using only_magnitude, chooses the gradient-based
                           method to apply directional signs (saliency or gradcam).
        n_epochs (int): Number of optimization steps for GNNExplainer.

    Returns:
        Tensor: Node-level importance scores (1D tensor).
    '''
    assert node_masking in ['object', 'attributes']
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=n_epochs),
        explanation_type=ExplanationType.model,
        model_config={
            'mode': ModelMode.regression,  # appropriate for both regression and binary classification
            'return_type': 'raw',          # work with the raw output of the model
            'task_level': 'graph',         # graph-level prediction task (we have a single output value)
        },
        node_mask_type=MaskType.object if node_masking == 'object' else MaskType.attributes,
        edge_mask_type=MaskType.object if isinstance(model, ModelEdgeGIN) else None
    )

    # the core algorithm: maximize information between submask and prediction under regularizing constraints
    assert data.batch is None  # this use case is handled (note that index=... is omitted below)
    explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    assert explanation.node_mask is not None

    # aggregate feature-level importances into a single score per node by taking the L2 norm
    importance_magnitudes = explanation.node_mask.norm(p=2, dim=1)  # shape: [n_nodes]

    if only_magnitude:
        # early exit if no directional information is necessary
        return importance_magnitudes

    # GNNExplainer maximizes mutual information, and as such does not indicate in what
    # direction it influences the output. Direction (sign) of importance may be taken
    # from a gradient-based method.
    match sign_source:
        case 'saliency':
            directional_importances = saliency_map(model, data, only_magnitude=False)
        case 'gradcam':
            directional_importances = gradcam_node_importances(model, data, only_magnitude=False)
        case _:
            raise NotImplementedError(f"Cannot compute directional node importances with '{sign_source}'.")
    importance_directions = torch.sign(directional_importances)

    directional_node_importances = importance_magnitudes * importance_directions
    return directional_node_importances
