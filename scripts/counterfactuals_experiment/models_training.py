from source import train_classification
import sys

def run_train(dataset,seed):
    sys.argv = [
        "train_classification.py",
        "--dataset", dataset,
        "--model_size", "512",
        "--dropout", "0.3",
        "--seed", str(seed)
    ]
    train_classification.main() 

# Run with multiple seeds
for dataset in ["CYP3A4_Veith", "CYP2D6_Veith", "hERG_Karim"]:
    print(f"\n=== Training on dataset {dataset} ===")
    for seed in [5, 15, 25]:
        print(f"\n=== Training with seed {seed} ===")
        run_train(dataset, seed)