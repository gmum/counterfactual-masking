import os
from tdc.single_pred import ADME, Tox
import wandb
import torch
from tqdm import trange, tqdm
import numpy as np
import torch.nn as nn
import random
import argparse
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


from source.models import GraphIsomorphismNetwork_classification
from source.featurizers.graphs import Featurizer2D
from source.dataset_cleaner import process_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CYP3A4_Veith", "CYP2D6_Veith", "hERG_Karim"])
    parser.add_argument("--model_size", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=False, default=15)
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    return parser.parse_args()

def get_data_module(name):
    if name == "hERG_Karim":
        return Tox(name)
    else:
        return ADME(name)

def calculate_metrics(targets, predictions, predictions_binary):
    accuracy = accuracy_score(targets, predictions_binary)
    precision = precision_score(targets, predictions_binary)
    recall = recall_score(targets, predictions_binary)
    f1 = f1_score(targets, predictions_binary)
    auc = roc_auc_score(targets, predictions)
    return accuracy, precision, recall, f1, auc

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    loss_data = []
    epoch_predictions = []
    epoch_targets = []

    for data in tqdm(train_loader):
        data = data.to(device)
        y = data.y
        optimizer.zero_grad()
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
    return loss_value, epoch_predictions, epoch_targets

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
    return val_loss_value, val_predictions, val_targets

def main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 300
    batch_size = args.batch_size

    os.makedirs(f"models/gin_{args.dataset.lower()}", exist_ok=True)

    data_module = get_data_module(args.dataset)
    split = data_module.get_split(seed = seed)
    
    featurizer = Featurizer2D('Y', 'Drug')
    data_dir = f"./results_{args.dataset}"
    split_data = featurizer(split=split, path=data_dir)

    # Remove salts and process datasets
    x_train = process_dataset(split_data['train'], featurizer)
    x_valid = process_dataset(split_data['valid'], featurizer)
    x_test = process_dataset(split_data['test'], featurizer)

    train_loader = GraphDataLoader(x_train, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(x_valid, batch_size=batch_size)
    test_loader = GraphDataLoader(x_test, batch_size=batch_size)
    
    model = GraphIsomorphismNetwork_classification(
        n_input_features=x_train[0].x.shape[1],
        hidden_size=args.model_size,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()
    # wandb.init(
    #     project=f"{args.dataset}_seeds",
    #     config={
    #         "dataset": args.dataset,
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "model_size": args.model_size,
    #         "dropout": args.dropout,
    #         "seed": args.seed
    #     },
    #     name=f"hidden_{args.model_size}_dropout_{args.dropout}_{args.seed}"
    # )

    best_val_loss = float('inf')
    best_path = f"models/gin_{args.dataset.lower()}/model_{args.dataset}_hidden_{args.model_size}_dropout_{args.dropout}_seed_{args.seed}.pth"
    wait = 0
    # Training loop
    for epoch in trange(1, epochs + 1, desc="Epochs"):
        train_loss, train_preds, train_targets = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_targets = validate(model, valid_loader, loss_fn, device)

        train_bin = [1 if p >= 0.5 else 0 for p in train_preds]
        val_bin = [1 if p >= 0.5 else 0 for p in val_preds]

        train_metrics = calculate_metrics(train_targets, train_preds, train_bin)
        val_metrics = calculate_metrics(val_targets, val_preds, val_bin)

        # Log
        # log_dict = {
        #     'epoch': epoch,
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        #     **{f"train_{m}": v for m, v in zip(['acc','prec','recall','f1','auc'], train_metrics)},
        #     **{f"val_{m}": v for m, v in zip(['acc','prec','recall','f1','auc'], val_metrics)}
        # }
        # wandb.log(log_dict)

        if val_loss < best_val_loss - 0.01:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            wait = 0
        else:
            wait += 1
            if wait > 20:
                print("Early stopping")
                break
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best model saved at: {best_path}")
    
    # Test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model.predict(model, batch).flatten()
            test_preds.extend(out.cpu().numpy())
            test_targets.extend(batch.y.flatten().cpu().numpy())

    test_bin = [1 if p >= 0.5 else 0 for p in test_preds]
    test_metrics = calculate_metrics(test_targets, test_preds, test_bin)
    print("Test Metrics:", test_metrics)
    # wandb.log({
    #     **{f"test_{m}": v for m, v in zip(['acc','prec','recall','f1','auc'], test_metrics)}
    # })
    # wandb.finish()


if __name__ == '__main__':
    main()

