from pathlib import Path
from typing import Literal, List, Dict

import numpy as np
import torch
import sklearn
from torch_geometric.loader import DataLoader as GraphDataLoader

from models import Model


class Trainer():

    def __init__(
        self,
        model: Model,
        model_file_path: Path,
        dataloaders: Dict[Literal['train', 'valid', 'test'], GraphDataLoader],
        device: torch.device,
        n_epochs: int,
        initial_lr: float,
        lr_scheduler_every_epochs: int,
        lr_scheduler_scaling_factor: float,
        max_epochs_without_improvement: int | None,
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.dataloaders = dataloaders
        self.model_file_path = model_file_path

        match model.task_type:
            case 'regression':
                self.loss_fn = torch.nn.SmoothL1Loss()
                self.loss_value_lower_is_better = True
            case 'classification':
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
                self.loss_value_lower_is_better = True
            case _:
                raise ValueError("Parameter task_type must be either 'regression' or 'classification'.")

        self.n_epochs = n_epochs
        self.max_epochs_without_improvement = (
            max_epochs_without_improvement if max_epochs_without_improvement is not None else
            n_epochs
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_scheduler_every_epochs,
            gamma=lr_scheduler_scaling_factor,
        )


    def _epoch(
        self,
        phase: Literal['train', 'valid'],
        dataloader,
    ):
        history: Dict[str, List[float]] = {
            'loss': [],
            'predictions': [],
            'targets': [],
        }

        if phase == 'train':
            self.model.train()
        elif phase == 'valid':
            self.model.eval()
            ctx = torch.inference_mode()
            ctx.__enter__()

        for data in dataloader:
            data = data.to(self.device)
            targets = data.y
            self.model.zero_grad()
            raw_predictions = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = self.loss_fn(raw_predictions.flatten(), targets.flatten())

            predictions = self.model.to_task_type_relevant_predictions(raw_predictions, binarize=False)
            history['loss'       ].append(loss       .cpu().detach().numpy())
            history['predictions'].extend(predictions.cpu().detach().numpy())
            history['targets'    ].extend(targets    .cpu().detach().numpy())

            if phase == 'train':
                loss.backward()
                self.optimizer.step()

        if phase == 'train':
            self.scheduler.step()
        elif phase == 'valid':
            ctx.__exit__(None, None, None)

        history = {
            name: np.array(lst)
            for name, lst in history.items()
        }
        match self.model.task_type:
            case 'classification':
                return {
                    'mean_loss': np.mean(history['loss']),
                    'roc_auc': sklearn.metrics.roc_auc_score(history['targets'], history['predictions'])
                }
            case 'regression' | _:
                return {
                    'mean_loss': np.mean(history['loss']),
                    'mae': np.mean(np.abs(history['predictions'] - history['targets'])),
                    'rmse': np.sqrt(np.mean((history['predictions'] - history['targets'])**2)),
                }


    def fit(self, verbose: bool = True):
        best_val_loss = float('inf' if self.loss_value_lower_is_better else '-inf')
        epoch_last_model_saved = 0
        for epoch_i in range(1, self.n_epochs + 1):
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            metrics = {
                phase: self._epoch(phase, self.dataloaders[phase])
                for phase in ('train', 'valid')
            }
            if dumping_to_file := (
                      metrics['valid']['mean_loss']*(+1 if self.loss_value_lower_is_better else -1)
                    < best_val_loss
                ):
                best_val_loss = metrics['valid']['mean_loss']
                torch.save(self.model.state_dict(), self.model_file_path)
                epoch_last_model_saved = epoch_i
            if epoch_i - epoch_last_model_saved > self.max_epochs_without_improvement:
                if verbose:
                    print(f"No improvement for {self.max_epochs_without_improvement} epochs "
                          "=> early exit.", flush=True)
                break
            if verbose:
                match self.model.task_type:
                    case 'classification':
                        print(
                            f"[Epoch {epoch_i:3d}/{self.n_epochs:3d}]  "
                            f"Train loss: {metrics['train']['mean_loss']:.5f}, "
                            f"ROC_AUC: {metrics['train']['roc_auc']:.5f}.  "
                            f"Validation loss: {metrics['valid']['mean_loss']:.5f}, "
                            f"ROC_AUC: {metrics['valid']['roc_auc']:.5f}.  "
                            f"LR: {lr:.2e}.{'  (SAVED.)' if dumping_to_file else ''}",
                            flush=True
                        )
                    case 'regression' | _:
                        print(
                            f"[Epoch {epoch_i:3d}/{self.n_epochs:3d}]  "
                            f"Train loss: {metrics['train']['mean_loss']:.5f}, "
                            f"MAE: {metrics['train']['mae']:.5f}, "
                            f"RMSE: {metrics['train']['rmse']:.5f}.  "
                            f"Validation loss: {metrics['valid']['mean_loss']:.5f}, "
                            f"MAE: {metrics['valid']['mae']:.5f}, "
                            f"RMSE: {metrics['valid']['rmse']:.5f}.  "
                            f"LR: {lr:.2e}.{'  (SAVED.)' if dumping_to_file else ''}",
                            flush=True
                        )
