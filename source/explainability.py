import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ExplanationType, ModelMode, MaskType


def contrastive_saliency_map(model, data):

    model.eval()

    data.x.requires_grad_()

    if data.x.grad is not None:
        data.x.grad.zero_()

    output = model.predict(model, data)

    output.backward()

    relu_grad = F.relu(data.x.grad)

    saliency_map = torch.norm(relu_grad, dim=1)

    data.x.requires_grad_(False)

    return saliency_map


class GradCAM:
    def __init__(self, model, target_layer_name='c3'):
        self.model = model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        target_layer = getattr(self.model, target_layer_name)
        target_layer.register_forward_hook(lambda _m, _i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_backward_hook(lambda _m, _gi, go: setattr(self, 'gradients', go[0].detach()))

    def node_importances(self, data, eps=1e-9):
        data.x.requires_grad = True

        self.model.eval()
        self.model.zero_grad()
        output = self.model.predict(self.model, data)
        output.backward()

        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (self.activations * weights).sum(dim=1)
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)

        data.x.requires_grad = False
        return cam


def gradcam_node_importances(model, data):
    gradcam = GradCAM(model)
    return gradcam.node_importances(data)


def gnnexplainer_node_importances(model, data, n_epochs=100, batch=None):
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=n_epochs),
        explanation_type=ExplanationType.model,
        model_config=dict(
            mode=ModelMode.regression,
            return_type='raw',
            task_level='graph',
        ),
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object,
    )
    explanation = explainer(data.x, data.edge_index, batch=batch)
    node_importance = explanation.node_mask.norm(p=2, dim=1)
    return node_importance
