import exmol
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import pickle
import gc
import argparse
import os

from scripts.counterfactuals_experiment.helpers_counterfactual import counterfactuals_metrics, similarity, fix_radicals

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()

def get_matching_folders(parent_dir, prefix):
    return [
        name for name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, name)) and name.startswith(prefix)
    ]

def print_summary(name, metrics):
    valid_counts = [c for c in metrics['count'] if c > 0]
    before_validity = [v for v, c in zip(metrics['before_validity'], metrics['before_count']) if c > 0]
    valid_validity = [v for v, c in zip(metrics['validity'], metrics['count']) if c > 0]
    valid_similarity = [s for s, c in zip(metrics['similarity'], metrics['count']) if c > 0]
    valid_diversity = [d for d, c in zip(metrics['diversity'], metrics['count']) if c > 0]
    valid_accessibility = [a for a, c in zip(metrics['accessibility'], metrics['count']) if c > 0]

    print(f"\n{name}")
    print(f"Before validity: {np.mean(before_validity):.4f}")
    print(f"Validity: {np.mean(valid_validity):.4f}")
    print(f"Similarity: {np.mean(valid_similarity):.4f}")
    print(f"Diversity: {np.mean(valid_diversity):.4f}")
    print(f"Accessibility: {np.mean(valid_accessibility):.4f}")
    print(f"Average number of counterfactuals: {np.mean(valid_counts):.2f}")
    print(f"Success rate: {np.mean([1 if c > 0 else 0 for c in metrics['count']]):.4f}")

def calculate_metrics(metrics):
    valid_counts = [c for c in metrics['count'] if c > 0]
    before_validity = [v for v, c in zip(metrics['before_validity'], metrics['before_count']) if c > 0]
    valid_validity = [v for v, c in zip(metrics['validity'], metrics['count']) if c > 0]
    valid_similarity = [s for s, c in zip(metrics['similarity'], metrics['count']) if c > 0]
    valid_diversity = [d for d, c in zip(metrics['diversity'], metrics['count']) if c > 0]
    valid_accessibility = [a for a, c in zip(metrics['accessibility'], metrics['count']) if c > 0]
    return {
        "before_validity": [float(np.mean(before_validity))],
        "validity": [float(np.mean(valid_validity))],
        "similarity": [float(np.mean(valid_similarity))],
        "diversity": [float(np.mean(valid_diversity))],
        "accessibility": [float(np.mean(valid_accessibility))],
        "average_count": [float(np.mean(valid_counts))],
        "success_rate": [float(np.mean([1 if c > 0 else 0 for c in metrics['count']]))]
    }

def print_summary_across_folders(name, metrics_dict):
    print(metrics_dict)
    print(f"\n{name}")
    keys = metrics_dict.keys()
    for key in keys:
        values = [v for v in metrics_dict[key] if isinstance(v, (int, float)) and not np.isnan(v)]
        print(f"{key}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}")
        
def merge_dicts_of_lists(d1, d2):
    merged = {}
    for key in set(d1) | set(d2):
        merged[key] = d1.get(key, []) + d2.get(key, [])
    return merged

def compute_score(candidate, pred, selected, org_smiles, org_pred, alpha=0.6):
        # Similarity to original molecule
        sim_to_org = similarity(org_smiles, candidate)

        # Diversity from already selected (penalize high similarity to selected)
        if not selected:
            diversity_penalty = 0
        else:
            diversity_penalty = 1 - np.mean([similarity(candidate, s) for s in selected])

        # Composite score: Higher is better
        score = (
            alpha * sim_to_org +
            (1 - alpha) * diversity_penalty
        )
        return score


def optimizer_counterfactuals(org_smiles, org_pred, counterfactuals_smiles, counterfactuals_predictions,
                                 number_of_counterfactuals=5, alpha=0.6):
    # Choose only examples which change class
    remaining_indexes = []
    for i, (smiles, pred) in enumerate(zip(counterfactuals_smiles, counterfactuals_predictions)):
        # Only keep counterfactuals that flip the class
        if (org_pred < 0.5 and pred > 0.5) or (org_pred > 0.5 and pred < 0.5):
            remaining_indexes.append(i)

    if len(remaining_indexes) < number_of_counterfactuals:
        return remaining_indexes
    
    # Canonicalize SMILES before checking for duplicates
    chosen_smiles = []
    for i in remaining_indexes:
        mol = Chem.MolFromSmiles(counterfactuals_smiles[i])
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
        else:
            canonical = counterfactuals_smiles[i]
        chosen_smiles.append(canonical)
    # Remove duplicate SMILES by keeping only the first occurrence
    unique_smiles = {}
    unique_indexes = []
    for idx, smi in zip(remaining_indexes, chosen_smiles):
        if smi not in unique_smiles:
            unique_smiles[smi] = idx
            unique_indexes.append(idx)
    remaining_indexes = unique_indexes

    # Iterative selection with full scoring
    selected_indexes = []
    selected_smiles = []

    while len(selected_indexes) < number_of_counterfactuals and remaining_indexes:
        scores = []
        for i in remaining_indexes:
            smiles = counterfactuals_smiles[i]
            pred = counterfactuals_predictions[i]
            score = compute_score(smiles, pred, selected_smiles, org_smiles, org_pred, alpha)
            scores.append(score)

        best_idx_in_remaining = np.argmax(scores)
        best_global_idx = remaining_indexes[best_idx_in_remaining]
        selected_indexes.append(best_global_idx)
        selected_smiles.append(counterfactuals_smiles[best_global_idx])
        del remaining_indexes[best_idx_in_remaining]
    return selected_indexes

# Helper function to check if a SMILES string is a single molecule
def is_single_molecule(smiles):
    return '.' not in smiles


def main():
    args = parse_args()
    folders = get_matching_folders(f"results_{args.dataset}", "result_counterfactuals")
    
    all_metrics = {
        'gnnexplainer_nodes': {},
        'exmol': {},
        'closer': {},
        'crem': {},
        'difflinker': {}
        }

    for folder in folders:
        pkl_files = get_matching_folders(folder, "counterfactual_predictions_")
        counterfactual_predictions_list = []
        for file in pkl_files:
            with open(file, 'rb') as f:
                counterfactual_predictions_list.append(pickle.load(f))

        counter_predictions = {}
        for d in counterfactual_predictions_list:
            for key, value in d.items():
                if key not in counter_predictions:
                    counter_predictions[key] = []
                counter_predictions[key].extend(value)
        results_files = get_matching_folders(folder, "counterfactual_results_")
        results_list = []
        for file in results_files:
            with open(file, 'rb') as f:
                results_list.append(pickle.load(f))

        counter_results = {}
        for d in results_list:
            for key, value in d.items():
                if key not in counter_results:
                    counter_results[key] = []
                counter_results[key].extend(value)

        metrics_gnnexplainer_nodes = {'before_validity': [], 'before_count': [], 'validity': [], 'similarity': [], 'diversity': [], 'accessibility': [] , 'count': []}
        metrics_closer = {'before_validity': [], 'before_count': [], 'validity': [], 'similarity': [], 'diversity': [], 'accessibility': [] , 'count': []}
        metrics_exmol = {'before_validity': [], 'before_count': [], 'validity': [], 'similarity': [], 'diversity': [], 'accessibility': [] , 'count': []}
        metrics_crem = {'before_validity': [], 'before_count': [], 'validity': [], 'similarity': [], 'diversity': [], 'accessibility': [] , 'count': []}
        metrics_difflinker = {'before_validity': [], 'before_count': [], 'validity': [], 'similarity': [], 'diversity': [], 'accessibility': [] , 'count': []}

        for index in tqdm(range(len(counter_results['org']))):
            # ExMol
            if len(counter_results['exmol'][index]) > 0:
                exmol_output = exmol.cf_explain(counter_results['exmol'][index])
                # Skip if no counterfactuals were generated
                if len(exmol_output) <= 1:
                    counterfactulas_exmol = []
                    counterfactulas_pred_exmol = []

                else:
                    counterfactulas_exmol = [sample.smiles for sample in exmol_output[1:]]  # Exclude original
                    counterfactulas_pred_exmol = [sample.yhat for sample in exmol_output[1:]] # Exclude original
                    counterfactulas_exmol_before = [sample.smiles for sample in counter_results['exmol'][index][1:]]  # Exclude original
                    counterfactulas_pred_exmol_before = [sample.yhat for sample in counter_results['exmol'][index][1:]] # Exclude original
            else:
                counterfactulas_exmol = []
                counterfactulas_pred_exmol = []
                counterfactulas_exmol_before = []
                counterfactulas_pred_exmol_before = []
            d, v, s, a = counterfactuals_metrics(counter_results["org"][index], counter_predictions["org"][index], counterfactulas_exmol_before, counterfactulas_pred_exmol_before)
            metrics_exmol['before_validity'].append(v)
            metrics_exmol['before_count'].append(len(counterfactulas_exmol_before))
            d, v, s, a = counterfactuals_metrics(counter_results["org"][index], counter_predictions["org"][index], counterfactulas_exmol, counterfactulas_pred_exmol)
            metrics_exmol['diversity'].append(d)
            metrics_exmol['validity'].append(v)
            metrics_exmol['similarity'].append(s)
            metrics_exmol['accessibility'].append(a)
            metrics_exmol['count'].append(len(counterfactulas_exmol))


            # GNNExplainer Nodes
            single_cf_indexes = [i for i, sample in enumerate(counter_results['gnnexplainer_nodes'][index]) if is_single_molecule(sample)]
            counter_results['gnnexplainer_nodes'][index] = [counter_results['gnnexplainer_nodes'][index][i] for i in single_cf_indexes]
            counter_predictions['gnnexplainer_nodes'][index] = [counter_predictions['gnnexplainer_nodes'][index][i] for i in single_cf_indexes]
            d, v, s, a = counterfactuals_metrics(counter_results['org'][index], counter_predictions['org'][index], counter_results['gnnexplainer_nodes'][index], counter_predictions['gnnexplainer_nodes'][index])
            metrics_gnnexplainer_nodes['before_validity'].append(v)
            metrics_gnnexplainer_nodes['diversity'].append(d)
            metrics_gnnexplainer_nodes['validity'].append(v)
            metrics_gnnexplainer_nodes['similarity'].append(s)
            metrics_gnnexplainer_nodes['accessibility'].append(a)
            metrics_gnnexplainer_nodes['count'].append(len(counter_results['gnnexplainer_nodes'][index]))

            # Closer (Nearest Neighbor)
            single_cf_indexes = [i for i, sample in enumerate(counter_results['closer'][index]) if is_single_molecule(sample)]
            counter_results['closer'][index] = [counter_results['closer'][index][i] for i in single_cf_indexes]
            counter_predictions['closer'][index] = [counter_predictions['closer'][index][i] for i in single_cf_indexes]
            d, v, s, a = counterfactuals_metrics(counter_results['org'][index], counter_predictions['org'][index], counter_results['closer'][index], counter_predictions['closer'][index])
            metrics_closer['before_validity'].append(v)
            metrics_closer['diversity'].append(d)
            metrics_closer['validity'].append(v)
            metrics_closer['similarity'].append(s)
            metrics_closer['accessibility'].append(a)
            metrics_closer['count'].append(len(counter_results['closer'][index]))

            # Crem
            single_cf_indexes = [i for i, sample in enumerate(counter_results['crem'][index]) if is_single_molecule(sample)]
            counter_results['crem'][index] = [counter_results['crem'][index][i] for i in single_cf_indexes]
            counter_predictions['crem'][index] = [counter_predictions['crem'][index][i] for i in single_cf_indexes]
            d, v, s, a = counterfactuals_metrics(counter_results['org'][index], counter_predictions['org'][index], counter_results['crem'][index], counter_predictions['crem'][index])
            metrics_crem['before_validity'].append(v)
            metrics_crem['before_count'].append(len(counter_results['crem'][index]))
            indexes = optimizer_counterfactuals(counter_results['org'][index], counter_predictions['org'][index], counter_results['crem'][index], counter_predictions['crem'][index], 3,  0.5)
            new_counterfactuals_crem_2 = [counter_results['crem'][index][idx] for idx in indexes]
            new_counterfactuals_pred_crem_2 = [counter_predictions['crem'][index][idx] for idx in indexes]
            d, v, s, a = counterfactuals_metrics(counter_results['org'][index], counter_predictions['org'][index], new_counterfactuals_crem_2, new_counterfactuals_pred_crem_2)
            metrics_crem['diversity'].append(d)
            metrics_crem['validity'].append(v)
            metrics_crem['similarity'].append(s)
            metrics_crem['accessibility'].append(a)
            metrics_crem['count'].append(len(new_counterfactuals_crem_2))

            # Difflinker
            single_cf_indexes = [i for i, sample in enumerate(counter_results['difflinker'][index]) if is_single_molecule(sample)]
            counter_results['difflinker'][index] = [counter_results['difflinker'][index][i] for i in single_cf_indexes]
            counter_predictions['difflinker'][index] = [counter_predictions['difflinker'][index][i] for i in single_cf_indexes]
            d, v, s, a = counterfactuals_metrics(counter_results['org'][index], counter_predictions['org'][index], counter_results['difflinker'][index], counter_predictions['difflinker'][index])
            metrics_difflinker['before_validity'].append(v)
            metrics_difflinker['before_count'].append(len(counter_results['difflinker'][index]))
            indexes = optimizer_counterfactuals(counter_results['org'][index], counter_predictions['org'][index], counter_results['difflinker'][index], counter_predictions['difflinker'][index], 3,  0.5)
            new_counterfactuals_difflinker = [Chem.MolToSmiles(fix_radicals(Chem.MolFromSmiles(counter_results['difflinker'][index][idx]))) for idx in indexes]
            new_counterfactuals_pred_difflinker = [counter_predictions['difflinker'][index][idx] for idx in indexes]
            d, v, s, a = counterfactuals_metrics(counter_results['org'][index], counter_predictions['org'][index], new_counterfactuals_difflinker, new_counterfactuals_pred_difflinker)
            metrics_difflinker['diversity'].append(d)
            metrics_difflinker['validity'].append(v)
            metrics_difflinker['similarity'].append(s)
            metrics_difflinker['accessibility'].append(a)
            metrics_difflinker['count'].append(len(new_counterfactuals_pred_difflinker))
            
        all_metrics['gnnexplainer_nodes'] = merge_dicts_of_lists(all_metrics['gnnexplainer_nodes'], calculate_metrics(metrics_gnnexplainer_nodes))
        all_metrics['exmol'] = merge_dicts_of_lists(all_metrics['exmol'], calculate_metrics(metrics_exmol))
        all_metrics['closer'] = merge_dicts_of_lists(all_metrics['closer'], calculate_metrics(metrics_closer))
        all_metrics['crem'] = merge_dicts_of_lists(all_metrics['crem'], calculate_metrics(metrics_crem))
        all_metrics['difflinker'] = merge_dicts_of_lists(all_metrics['difflinker'], calculate_metrics(metrics_difflinker))

        print_summary("gnnexplainer_nodes",metrics_gnnexplainer_nodes)
        print_summary("exmol",metrics_exmol)
        print_summary("closer",metrics_closer)
        print_summary("crem",metrics_crem)
        print_summary("difflinker",metrics_difflinker)
        with open(f"{folder}/all_metrics.pkl", "wb") as f:
            pickle.dump(all_metrics, f)

        del counterfactual_predictions_list
        del results_list
        del counter_predictions
        del counter_results
        gc.collect()

    for method_name, metrics_list in all_metrics.items():
        print_summary_across_folders(method_name, metrics_list)

if __name__ == "__main__":
    main()