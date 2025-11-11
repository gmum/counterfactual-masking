import os
from tdc.single_pred import ADME
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import trange, tqdm
import numpy as np
import random
import argparse
import wandb

from source.models import GraphIsomorphismNetwork
from source.featurizers.graphs import Featurizer2D
from source.dataset_cleaner import process_dataset

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    loss_data = []
    epoch_predictions = []
    epoch_targets = []

    for data in tqdm(train_loader):
        data = data.to(device)
        y = data.y
        model.zero_grad()
        preds = model.predict(model, data)

        preds = preds.flatten()
        y = y.flatten()

        loss = loss_fn(preds, y)
        loss_data.append(loss.cpu().detach().numpy())

        epoch_predictions.extend(preds.cpu().detach().numpy())
        epoch_targets.extend(y.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

    loss_value = np.mean(loss_data)
    epoch_mae = np.mean(np.abs(np.array(epoch_predictions) - np.array(epoch_targets)))
    epoch_rmse = np.sqrt(np.mean((np.array(epoch_predictions) - np.array(epoch_targets))**2))

    return loss_value, epoch_mae, epoch_rmse

def validate(model, valid_loader, loss_fn, device):
    model.eval()
    val_loss_data = []
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            y = data.y
            preds = model.predict(model, data)

            preds = preds.flatten()
            y = y.flatten()

            val_loss = loss_fn(preds, y)
            val_loss_data.append(val_loss.cpu().detach().numpy())

            val_predictions.extend(preds.cpu().detach().numpy())
            val_targets.extend(y.cpu().detach().numpy())

    val_loss_value = np.mean(val_loss_data)
    val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_targets)))
    val_rmse = np.sqrt(np.mean((np.array(val_predictions) - np.array(val_targets))**2))

    return val_loss_value, val_mae, val_rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=False, default=15)
    parser.add_argument("--dropout", type=float, required=False, default=0.15)
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    parser.add_argument("--path", type=str, required=False, default="checkpoints/gin")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = "Solubility_AqSolDB"
    epochs = 300
    batch_size = args.batch_size
    best_val_loss = float('inf')

    # wandb.init(
    #     project="graph-isomorphism-network",
    #     config={
    #         "dataset": dataset,
    #         "epochs": epochs,
    #         "batch_size": batch_size
    #     },
    #     name=f"hidden_{args.model_size}_dropout_{args.dropout}_{args.seed}"
    # )

    data = ADME(dataset)
    split = data.get_split(seed=seed)

    featurizer = Featurizer2D('Y', 'Drug')
    split_data = featurizer(
        split=split,
        path=f'./data/splits/{dataset}',
    )

    x_train = split_data["train"]
    x_valid = split_data["valid"]
    x_test = split_data["test"]

    # Remove salts and process datasets
    x_test = process_dataset(x_test, featurizer=featurizer)
    x_train = process_dataset(x_train, featurizer=featurizer)
    x_valid = process_dataset(x_valid, featurizer=featurizer)


    train_loader = GraphDataLoader(x_train, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(x_valid, batch_size=batch_size)
    test_loader = GraphDataLoader(x_test, batch_size=batch_size)

    model = GraphIsomorphismNetwork(n_input_features=22, hidden_size=args.model_size, dropout=args.dropout)
    model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in trange(1, epochs + 1):
        train_loss, train_mae, train_rmse = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mae, val_rmse = validate(model, valid_loader, loss_fn, device)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Train MAE: {train_mae:.5f}, Train RMSE: {train_rmse:.5f}, Val Loss: {val_loss:.5f}, Val MAE: {val_mae:.5f}, Val RMSE: {val_rmse:.5f}")

        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": train_loss,
        #     "train_mae": train_mae,
        #     "train_rmse": train_rmse,
        #     "val_loss": val_loss,
        #     "val_mae": val_mae,
        #     "val_rmse": val_rmse
        # })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.path}/model_trained_without_salts_hidden_{args.model_size}_dropout_{args.dropout}_seed_{args.seed}.pth")

    # wandb.finish()

if __name__ == '__main__':
    main()