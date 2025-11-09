# Enhancing Chemical Explainability Through Counterfactual Masking

![Masking Example](images/counterfactual_masking.png)

## Setup
The project uses **Python 3.11.5**.
```bash
git clone --branch main --recurse-submodules git@github.com:gmum/counterfactual-masking.git

cd counterfactual-masking/DiffLinker
# Download the DiffLinker model checkpoint
mkdir -p models
wget "https://zenodo.org/record/7121300/files/zinc_difflinker_given_anchors.ckpt?download=1" -O models/zinc_difflinker_given_anchors.ckpt
cd ../

# Download and extract the CReM dataset (ChEMBL22)
mkdir data
wget "https://www.qsar4u.com/files/cremdb/chembl22_sa2.db.gz" -O data/chembl22_sa2.db.gz
gunzip data/chembl22_sa2.db.gz

# Libraries
pip install torch==2.5.1+cu124 --index-url "https://download.pytorch.org/whl/cu124"
pip install -r requirements.txt

# DiffLinker
pip install -e .
```

## Dataset: Common Substructure Pair

**Note:** A preprocessed and filtered version of the Common Substructure Pair dataset is included in the repository and can be found in the `data_pubchem` directory.

### Regenerate the Dataset

To regenerate the dataset:

```bash
python -m scripts.superstructures.fetch_superstructures --output_folder data_pubchem --dataset_name common_substructure_pair_dataset
```

This script fetches superstructures from PubChem, processes them, and saves the results in the specified directory.

## Pairs Experiment
This experiment evaluates different masking strategies over pairs of molecules that share common substructures.

### Step 1: Train a Model

```bash
python -m source.train --model_size 512 --dropout 0.3 --batch_size 64 --seed 15
```

### Step 2: Run Pairs Experiment 

* **Single anchor**

```bash
python -m scripts.pairs_experiment.pairs_prediction --output_folder single_anchor_output  --model_path gin/model_trained_without_salts_hidden_512_dropout_0.3_seed_15.pth --pairs_dataset data_pubchem/common_substructure_pair_dataset.json --size_model 512 --same_anchors --number_of_anchors 1
```

* **Multiple anchors**

```bash
python -m scripts.pairs_experiment.pairs_prediction --output_folder 2_or_more_anchors_output  --model_path gin/model_trained_without_salts_hidden_512_dropout_0.3_seed_15.pth --pairs_dataset data_pubchem/common_substructure_pair_dataset.json --size_model 512 --number_of_anchors 2 --same_anchors
```

* **No anchor restrictions (Both variants)**
```bash
python -m scripts.pairs_experiment.pairs_prediction --output_folder no_restrictions_output  --model_path gin/model_trained_without_salts_hidden_512_dropout_0.3_seed_15.pth --pairs_dataset data_pubchem/common_substructure_pair_dataset.json --size_model 512 --same_anchors
```

### Step 3: View the Results
Open the following notebook to visualize results:
```bash
scripts/pairs_experiment/results_visualization_masking_evaluation.ipynb
```

## Counterfactuals Experiment

This experiment evaluates different counterfactual generation methods.

### Step 1: Train Models

```bash
python -m scripts.counterfactuals_experiment.models_training
```

### Step 2: Run Counterfactuals Experiment 
```bash
python -m scripts.counterfactuals_experiment.counterfactuals_generation --model_size 512  --seed <SEED> --dataset <DATASET> --model_path <MODEL_PATH>
```
### Parameters
| Argument        | Description                       | Used Values                                                                 |
|----------------|-----------------------------------|----------------------------------------------------------------------------------------|
| `--model_size` | Hidden size of the model          | `512`                                                                                  |
| `--seed`       | Random seed                       | `5`, `15`, `25`                                                                        |
| `--dataset`    | Name of the dataset               | `CYP3A4_Veith`, `CYP2D6_Veith`, `hERG`                                                 |
| `--model_path` | Path to the trained model file    | e.g., `models/gin_cyp2d6_veith/model_CYP2D6_Veith_hidden_512_dropout_0.3_seed_15.pth`        |


### Step 3: View the Results

```bash
python -m scripts.counterfactuals_experiment.counterfactuals_results_reader --dataset <DATASET>
```
### Parameters
| Argument        | Description                       | Used Values                                                                 |
|----------------|-----------------------------------|----------------------------------------------------------------------------------------|
| `--dataset`    | Name of the dataset               | `CYP3A4_Veith`, `CYP2D6_Veith`, `hERG`                                                 |


## Explainers Experiment

### Step 1: Train Models

```bash
(cd scripts/explainers_experiment && python step_01_training.py)
```
Training parameters are defined in `scripts/explainers_experiment/config.yaml`.

### Step 2: Run Explainers Experiment 
```bash
(cd scripts/explainers_experument && python step_02_masking.py)
```
Experiment parameters are defined in `scripts/explainers_experiment/config.yaml`.

### Step 3: Summmary of results (Table 3)
```bash
(cd scripts/explainers_experiment && jupyter execute step_03_summary.ipynb)
```
