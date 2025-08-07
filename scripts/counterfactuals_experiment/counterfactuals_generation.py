import argparse
import os
import pickle
import torch
from rdkit import Chem
from tdc.single_pred import ADME, Tox
from tqdm import tqdm
import exmol

from scripts.counterfactuals_experiment.helpers_counterfactual import smiles_predictions, get_ring_atom_indices, find_closest_counterfactual, PseudoModel
from source.featurizers.graphs import Featurizer2D
from source.models import GraphIsomorphismNetwork_classification
from source.explainability import gradcam_node_importances, gnnexplainer_node_importances
from source.linksGenerator import crem_fragment_replacement, diffLinker_fragment_replacement
from source.dataset_cleaner import process_dataset  
from source.linksGenerator import atomDeleter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CYP3A4_Veith", "CYP2D6_Veith", "hERG"])
    parser.add_argument("--model_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=False, default=15)
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()

def load_model(model_path, device):
    model = GraphIsomorphismNetwork_classification(n_input_features=22, hidden_size=int(512))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model 

def get_data_module(name: str):
    if name.lower() == "herg":
        return Tox(name)
    else:
        return ADME(name)
    
def main():
    args = parse_args()
    data_module = get_data_module(args.dataset)
    device = torch.device("cpu")
    featurizer = Featurizer2D('Y', 'Drug')
    data_dir = f"./results_{args.dataset}"
    split = data_module.get_split()
    split_data = featurizer(split=split, path=data_dir)
    x_test = process_dataset(split_data['test'], featurizer)
    x_train = process_dataset(split_data['train'], featurizer)
    model = load_model(args.model_path, device)
    model.to(device).eval()

    save_path = f"results_{args.dataset}/result_counterfactuals_{args.seed}"

    os.makedirs(save_path, exist_ok=True)

    # It was needed for the EXMOL
    model_pseudo = PseudoModel(model, featurizer, "cpu")


    save_results = {'org': [], 'exmol': [], 'gnnexplainer_nodes': [], 'closer': [], 'crem': [], 'difflinker': []}
    save_predictions = {'org': [], 'exmol': [], 'gnnexplainer_nodes': [], 'closer': [], 'crem': [], 'difflinker': []}
    
    x_test_conterfactual = x_test
    for i in tqdm(range(len(x_test_conterfactual))):
        org_smiles = x_test_conterfactual[i].smiles
        org_pred = model_pseudo(org_smiles)

        save_results['org'].append(org_smiles)
        save_predictions['org'].append(org_pred)
        
        # EXMOL
        # Generate counterfactuals
        print("Generating counterfactuals with EXMOL")
        try:
          exmol_output = exmol.sample_space(org_smiles, model_pseudo, batched=False)
          save_results['exmol'].append(exmol_output)
        except Exception as e:
          print(f"Error generating counterfactuals with EXMOL: {e}")
          save_results['exmol'].append([])

        #GNNExplainer
        atoms_importance = gnnexplainer_node_importances(model, x_test_conterfactual[i])
        # Top 10% of atoms
        important_atoms = sorted(range(len(atoms_importance)), key=lambda k: atoms_importance[k], reverse=True)
        top_atoms = important_atoms[:max(1, len(important_atoms) // 10)]
        
        # GNNExplainer nodes
        counterfactuals_gnnexplainer = [Chem.MolToSmiles(atomDeleter(Chem.MolFromSmiles(org_smiles), top_atoms))]
        counterfactuals_pred_gnnexplainer = smiles_predictions(model, featurizer, counterfactuals_gnnexplainer)
        none_indices = [idx for idx, pred in enumerate(counterfactuals_pred_gnnexplainer) if pred is None]
        counterfactuals_gnnexplainer = [cf for idx, cf in enumerate(counterfactuals_gnnexplainer) if idx not in none_indices]
        counterfactuals_pred_gnnexplainer = [pred for idx, pred in enumerate(counterfactuals_pred_gnnexplainer) if idx not in none_indices]
        save_results['gnnexplainer_nodes'].append(counterfactuals_gnnexplainer)
        save_predictions['gnnexplainer_nodes'].append(counterfactuals_pred_gnnexplainer)
        
        # Find closest counterfactuals
        closest_counterfactuals = find_closest_counterfactual(org_smiles, x_train, model_pseudo, number_of_counterfactuals=3)
        save_results['closer'].append([ex[0] for ex in closest_counterfactuals])
        save_predictions['closer'].append([model_pseudo(smiles) for smiles, _, _ in closest_counterfactuals])

        # Ours
        # Atoms importance
        atoms_importance = gradcam_node_importances(model, x_test_conterfactual[i])
        # Top 20% of atoms
        important_atoms = sorted(range(len(atoms_importance)), key=lambda k: atoms_importance[k], reverse=True)
        
        top_atoms = important_atoms[:max(1, len(important_atoms) // 5)]
        atoms = top_atoms.copy()
        atoms.extend(get_ring_atom_indices(org_smiles, top_atoms))
        atoms = set(atoms)
        
        # Generate counterfactuals CReM
        print("Generating counterfactuals with CReM")
        counterfactuals_crem = crem_fragment_replacement(Chem.MolFromSmiles(org_smiles), atoms)
        counterfactuals_pred_crem = smiles_predictions(model, featurizer, counterfactuals_crem)
        # Check indices where counterfactuals_pred_crem is None
        none_indices = [idx for idx, pred in enumerate(counterfactuals_pred_crem) if pred is None]
        counterfactuals_crem = [cf for idx, cf in enumerate(counterfactuals_crem) if idx not in none_indices]
        counterfactuals_pred_crem = [pred for idx, pred in enumerate(counterfactuals_pred_crem) if idx not in none_indices]
        save_results['crem'].append(counterfactuals_crem)
        save_predictions['crem'].append(counterfactuals_pred_crem)

        # Generate counterfactuals DiffLinker
        print("Generating counterfactuals with DiffLinker")
        number_of_times = 0
        counterfactuals_difflinker = []
        while len(counterfactuals_difflinker) < 5 and number_of_times < 3:
            try:
              counterfactuals_difflinker.extend(diffLinker_fragment_replacement(Chem.MolFromSmiles(org_smiles), top_atoms, f"test_{args.dataset}_"))
            except Exception as e:
              print(f"Error generating counterfactuals with DiffLinker: {e}")
            number_of_times += 1
        counterfactuals_pred_difflinker = smiles_predictions(model, featurizer, counterfactuals_difflinker)
        none_indices = [idx for idx, pred in enumerate(counterfactuals_pred_difflinker) if pred is None]
        counterfactuals_difflinker = [cf for idx, cf in enumerate(counterfactuals_difflinker) if idx not in none_indices]
        counterfactuals_pred_difflinker = [pred for idx, pred in enumerate(counterfactuals_pred_difflinker) if idx not in none_indices]
        save_results['difflinker'].append(counterfactuals_difflinker)
        save_predictions['difflinker'].append(counterfactuals_pred_difflinker)


        if (i + 1) % 10 == 0 or (i + 1) == len(x_test_conterfactual):
            with open(os.path.join(save_path,f"counterfactual_results_{i + 1}.pkl"), "wb") as f:
                pickle.dump(save_results, f)
            with open(os.path.join(save_path,f"counterfactual_predictions_{i + 1}.pkl"), "wb") as f:
                pickle.dump(save_predictions, f)
            save_results = {'org': [], 'exmol': [], 'gnnexplainer_nodes': [], 'closer': [], 'crem': [], 'difflinker': []}
            save_predictions = {'org': [], 'exmol': [], 'gnnexplainer_nodes': [], 'closer': [], 'crem': [], 'difflinker': []}

if __name__ == "__main__":
    main()