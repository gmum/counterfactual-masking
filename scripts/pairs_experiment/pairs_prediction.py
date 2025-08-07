import os
import gc
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from rdkit import Chem

from source.featurizers.graphs import Featurizer2D
from source.models import GraphIsomorphismNetwork
from source.changer import feature_zeroing
from source.linksGenerator import diffLinker_fragment_replacement, crem_fragment_replacement
from source.dataset_cleaner import process_pairs_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--size_model", type=str, required=True)
    parser.add_argument("--pairs_dataset", type=str, required=True)
    parser.add_argument("--max_replacements", type=int, default=None)
    parser.add_argument("--number_of_anchors", type=int, default=None, 
        help="Specify the number of anchors for the masking process. "
             "Use None for no restrictions, 1 for exactly one anchor, "
             "or any integer greater than 1 for two or more anchors."
    )
    parser.add_argument("--same_anchors", action="store_true")
    parser.add_argument("--save_all_generated", action="store_true")
    return parser.parse_args()

def load_model(model_path, size_model, device):
    model = GraphIsomorphismNetwork(n_input_features=22, hidden_size=int(size_model))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def save_batch(output_folder, batch_id, embeddings, targets, output):
    np.save(f"{output_folder}/embeddings_batch_{batch_id}.npy", embeddings)
    np.save(f"{output_folder}/targets_batch_{batch_id}.npy", np.array(targets, dtype = object))
    with open(f"{output_folder}/predictions_batch_{batch_id}.pkl", "wb") as f:
        pickle.dump(output, f)

def make_prediction(index_mol, row, featurizer, model, batch_size, device, max_replacements, save_all_generated = False):
    embeddings = []
    target = []
    common_smile = row[0]
    molecules = row[1]
    predictions = {}
    with torch.no_grad():
        for index, smiles in enumerate(molecules):
            print("Orginal")
            data = featurizer.process_single(Chem.MolFromSmiles(smiles), 0, smiles)
            data = data.to(device)
            prediction = model.predict(model, data).item()
            out = model.predict(model, data, return_representations=True)
            embeddings.extend(out.cpu())
            target.extend(data.y.cpu().numpy()*0 + 1)
            if "Org" not in predictions:
                predictions["Org"] = [prediction]
            else:
                predictions["Org"].append(prediction)
            print("Zero Features")
            mol = Chem.MolFromSmiles(smiles)
            common_atoms = mol.GetSubstructMatch(Chem.MolFromSmiles(common_smile))
            all_atoms = list(range(mol.GetNumAtoms()))
            non_common_atoms = [atom for atom in all_atoms if atom not in common_atoms]
            mol_with_mask = feature_zeroing(data, non_common_atoms)
            masked_pred = model.predict(model, mol_with_mask).item()
            out = model.predict(model, mol_with_mask, return_representations=True)
            embeddings.extend(out.cpu())
            target.extend(data.y.cpu().numpy()*0 + 2)
            if "Masked" not in predictions:
                predictions["Masked"] = [masked_pred]
            else:
                predictions["Masked"].append(masked_pred)

            print("Masking Crem")
            crem_output = []
            grow_mols = crem_fragment_replacement(mol, non_common_atoms, max_replacements= max_replacements)
            print(f"Number of generated molecules: {len(grow_mols)}")
            mol_set = []
            for growed_mol in grow_mols:
                mol_gen = Chem.MolFromSmiles(growed_mol)
                if mol_gen is None:
                    print(f"Error: Failed to generate molecule from SMILES {growed_mol}")
                    continue
                if len(Chem.GetMolFrags(mol_gen)) > 1:
                    print(f"Error: Generated molecule {growed_mol} has more than one fragment")
                    continue
                mol_gen_data = featurizer.process_single(mol_gen, 1)
                mol_set.append(mol_gen_data)
            mol_loader = GraphDataLoader(mol_set, batch_size=batch_size, shuffle=False)
            for data in tqdm(mol_loader):
                data = data.to(device)
                out = model.predict(model, data, return_representations=True)
                embeddings.extend(out.cpu())
                target.extend(data.y.cpu().numpy()*0 + 3)
                preds = model.predict(model, data).flatten()
                crem_output.extend(preds.cpu().numpy())
                del data, out, preds
                torch.cuda.empty_cache()
                gc.collect()    
            print("Predicted")
            predictions[f"Crem_{index}"] = crem_output
            
            print("Masking DiffLinker")
            diff_output = []
            output_folder = None
            number_of_times = 0
            while len(diff_output) < 5 and number_of_times < 3:
                try:
                    if save_all_generated:
                        diffLinker_fragment_replacement(mol, non_common_atoms, f"pairs_model/output_{index_mol}_{index}")
                        output_folder = f"data_diff/pairs_model/output_{index_mol}_{index}/output"
                    else:
                        diffLinker_fragment_replacement(mol, non_common_atoms, f"pairs_model/output_{index}")
                        output_folder = f"data_diff/pairs_model/output_{index}/output"
                    for file_name in os.listdir(output_folder):
                        if file_name.endswith(".sdf"):
                            mol_gen = Chem.MolFromMolFile(os.path.join(output_folder, file_name))
                            if mol_gen is None or len(Chem.GetMolFrags(mol_gen)) > 1:
                                    continue
                            mol_gen_data = featurizer.process_single(mol_gen, 1)
                            mol_gen_data.to(device)
                            diff_output.append(model.predict(model, mol_gen_data).item())
                            out = model.predict(model, mol_gen_data, return_representations=True)
                            embeddings.extend(out.cpu())
                            target.extend(mol_gen_data.y.cpu().numpy()*0 + 4)
                            del mol_gen_data, out
                            torch.cuda.empty_cache()
                            gc.collect()
                except Exception as e:
                    print(f"Error processing DiffLinker molecules: {e}")
                    number_of_times += 1
                    continue
                number_of_times += 1
            predictions[f"Diff_{index}"] = diff_output
    return predictions, embeddings, target

def main():
    args = parse_args()
    
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featurizer = Featurizer2D('Y', 'Drug')
    model = load_model(args.model_path, args.size_model, device)

    embeddings = []
    targets = []
    model.eval()

    pairs_array = process_pairs_dataset(args.pairs_dataset, model, featurizer, device, args.number_of_anchors, args.same_anchors)
    
    output = []
    for idx, row in tqdm(enumerate(pairs_array), total=len(pairs_array), desc="Processing pairs"):
        try:
            pred, embed, targ = make_prediction(idx, row, featurizer, model, batch_size, device, args.max_replacements, args.save_all_generated)
            embeddings.extend(embed)
            targets.extend(targ)
            output.append(pred)

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            output.append({"Org": [], "Masked": [], "Crem_0": [], "Crem_1": [], "Diff_0": [], "Diff_1": []})
            continue

        # Save batch every 5 iterations
        if (idx + 1) % 5 == 0 or idx == len(pairs_array) - 1:
            batch_id = idx // 5
            save_batch(output_folder, batch_id, embeddings, targets, output)
            output = []
            embeddings = []
            targets = []

            torch.cuda.empty_cache()
            gc.collect()

    idx = len(pairs_array) - 1
    batch_id = idx // 5
    if len(output) > 0:
        save_batch(output_folder, batch_id, embeddings, targets, output)

    print("Processing complete. Outputs saved to:", output_folder)

if __name__ == "__main__":
    main()

