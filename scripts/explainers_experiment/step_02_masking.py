#!/usr/bin/env python
# coding: utf-8

# serial run (all tasks):
# python3 step_01_masking.py '{"todo_index": -1,  "todos_reversed": 0,  "slice_start": 0,  "slice_stop": -1,  "log_file_name": "masking.log"}'

# parallel run (single task of index ITER, 0 <= ITER < 450):
# python3 step_01_masking.py '{"todo_index": ITER,  "todos_reversed": 0,  "slice_start": 0,  "slice_stop": 20,  "log_file_name": "masking-ITER.log"}'

import copy
import sys
from pathlib import Path
from functools import partial
from typing import Tuple, List, Dict, Literal, Callable
import tempfile
import json
import click
from loguru import logger
import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from rdkit import Chem
from models import load_model, Model
from featurizer import Featurizer
from explainability import (
    saliency_map,
    integrated_gradients,
    gradcam_node_importances,
    gnnexplainer_node_importances,
)
from common import (
    load_config,
    get_nested,
    select_device,
    set_random_seed,
)

sys.path.append('../..')
sys.path.append('../../DiffLinker')
from source.helpers import get_extended_ring_atom_indices
from source.linksGenerator import (
    diffLinker_fragment_replacement,
    crem_fragment_replacement,
)


torch.serialization.add_safe_globals([
    torch_geometric.data.data.Data,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.storage.GlobalStorage,
])



def load_workspace(config: dict) -> Tuple[Dict[str, Model], Dict[str, List[Data]], Dict]:

    def from_config(*args, **kwargs):
        return get_nested(config, *args, **kwargs)

    device = from_config('MASKING', 'device')
    output_folder_path = Path.cwd()/from_config('output_folder_name')

    todos_keys = ['dataset_name', 'split_index', 'model_name', 'explanation_method']  #, 'masking_method']
    todos_list = []

    models, datasets = {}, {}

    for dataset_name, task_type in [
        (dataset_name, dataset_properties['task_type'])
        for dataset_problem_type in ['ADME_DATASETS', 'TOX_DATASETS']
        for dataset_name, dataset_properties in from_config('DATA', dataset_problem_type).items()
    ]:
        for split_index in range(from_config('DATA', 'SPLIT', 'count')):
            test_dataset_file_name = f"dataset_{dataset_name}--split_{split_index}--partition_test.pth"
            test_dataset_file_path = output_folder_path / test_dataset_file_name
            test_dataset = torch.load(test_dataset_file_path, weights_only=True)
            datasets[(dataset_name, split_index)] = test_dataset

            for model_name, model_settings in from_config('MODEL', 'VARIANTS').items():
                model_arch = model_settings['architecture']
                model_file_name = f"dataset_{dataset_name}--split_{split_index}--model_{model_name}.pth"
                model_file_path = output_folder_path / model_file_name
                model_params = model_settings | Featurizer.feature_counts_dict()
                model = load_model(model_arch, task_type, model_file_path, device, **model_params)
                models[(dataset_name, split_index, model_name)] = model
                logger.info(
                    f"Loaded test dataset and model {dataset_name}/{split_index}/{model_name} "
                    f"({model.num_trainable_parameters()} parameters)"
                )
                for explanation_method in from_config('EXPLANATION', 'methods'):  # pylint: disable=possibly-unused-variable
                    #for masking_method in from_config('MASKING', 'methods'):
                    todos_list.append({k: locals()[k] for k in todos_keys})

    todos = [
        todo_series.to_dict()
        for _, todo_series in pd.DataFrame(todos_list).iterrows()
    ]
    logger.info(f"All todos: {len(todos)}")

    return models, datasets, todos


def mask_atoms(
    molecule: Data,
    atom_indices: List[int],
    method: Literal[
        'feature_zeroing',
        'feature_zeroing_rings',
        'counterfactual_difflinker',
        'counterfactual_difflinker_rings',
        'counterfactual_crem',
        'counterfactual_crem_rings',
    ],
    direction: Literal['+', '-'],
    atom_importances,
    explanation_method_fn: Callable,
    #is_atom_importance_random: bool,
    device: torch.device,
) -> List[torch_geometric.data.data.Data]:

    assert len(atom_indices) > 0
    assert len(atom_importances) > 0
    assert molecule.smiles == Chem.CanonSmiles(molecule.smiles)


    def _is_atom_importance_change_proper(
        new_molecule: Chem.rdchem.Mol,
        new_atom_indices: List[int]
    ) -> bool:
        '''
        Is the molecule modification due to masking expected to change the property
        in the right direction according to the explainer?
        '''
        #if is_atom_importance_random:
        #    # random importance assignment is effectively useless to decide whether
        #    # new atoms have the expected impact => accept molecule unconditionally
        #    return True

        # original molecule
        assert len(molecule.x) == len(atom_importances)
        influence_original = atom_importances[np.array(atom_indices)].mean()

        # new molecule (masked generatively or by feature zeroing)
        new_molecule.to(device)
        new_atom_importances = explanation_method_fn(data=new_molecule)  # new atom_influences
        if isinstance(new_atom_importances, torch.Tensor):
            new_atom_importances = new_atom_importances.cpu().numpy()
        influence_new = new_atom_importances[np.array(new_atom_indices)].mean()

        # assessment
        if direction == '+':
            # a decrease is expected in the masked molecule
            return influence_new < influence_original
        else:
            # an increase is expected in the masked molecule
            return influence_new > influence_original


    def _annotate_whether_proper(
        new_molecules_and_atom_indices: List[Tuple[Chem.rdchem.Mol | torch_geometric.data.Data, List[int]]]
    ) -> List[Tuple[torch_geometric.data.Data, bool]]:
        featurizer = Featurizer()
        annotated = []
        for new_molecule, new_atom_indices in new_molecules_and_atom_indices:
            if (
                not method.startswith('feature_zeroing')  # always true for zeroing
                and (new_molecule_smiles := Chem.CanonSmiles(Chem.MolToSmiles(new_molecule))) == molecule.smiles
            ):
                continue  # skip the new molecule if indistinguishable from the original one

            if method.startswith('feature_zeroing'):
                assert isinstance(new_molecule, torch_geometric.data.Data)
                featurized_new_molecule = new_molecule  # do not re-featurize
            else:
                assert isinstance(new_molecule, Chem.rdchem.Mol)
                featurized_new_molecule = featurizer(new_molecule, y=np.nan, smiles=new_molecule_smiles)

            is_change_proper = _is_atom_importance_change_proper(featurized_new_molecule, new_atom_indices)
            annotated.append((featurized_new_molecule, is_change_proper))

        return annotated


    def _of_unique_smileses(
        new_molecules_and_atom_indices: List[Tuple[Chem.rdchem.Mol, List[int]]]
    ) -> List[Tuple[Chem.rdchem.Mol, List[int]]]:
        unique, seen_smileses = [], set()
        for mol, ixs in new_molecules_and_atom_indices:
            if (mol_smi := Chem.CanonSmiles(Chem.MolToSmiles(mol))) not in seen_smileses:
                seen_smileses.add(mol_smi)
                unique.append((mol, ixs))
        logger.debug(f"Dedup: {len(unique)} / {len(new_molecules_and_atom_indices)} molecules unique.")
        return unique


    match method:
        case 'feature_zeroing' | 'feature_zeroing_rings':
            molecule_masked = molecule.clone()
            molecule_masked.x = molecule.x.clone()
            indices = get_extended_ring_atom_indices(molecule, atom_indices) if 'rings' in method else atom_indices
            molecule_masked.x[np.array(indices)] = 0.
            generated_molecules = [(molecule_masked, indices)]

        case 'counterfactual_difflinker' | 'counterfactual_difflinker_rings':
            indices = get_extended_ring_atom_indices(molecule, atom_indices) if 'rings' in method else atom_indices
            with tempfile.TemporaryDirectory(prefix="difflinker_", delete=True) as temp_folder:
                try:
                    all_generated_molecules = diffLinker_fragment_replacement(
                        Chem.MolFromSmiles(molecule.smiles),
                        indices,
                        folder=temp_folder,
                        return_new_atom_indices=True
                    )
                    generated_molecules = _of_unique_smileses(all_generated_molecules)
                except:
                    logger.warning(f"DiffLinker failed for {molecule.smiles}.")
                    generated_molecules = []
        case 'counterfactual_crem' | 'counterfactual_crem_rings':
            indices = get_extended_ring_atom_indices(molecule, atom_indices) if 'rings' in method else atom_indices
            try:
                all_generated_molecules = crem_fragment_replacement(
                    Chem.MolFromSmiles(molecule.smiles),
                    indices,
                    return_new_atom_indices=True
                )
                generated_molecules = _of_unique_smileses(all_generated_molecules)
            except:
                logger.warning(f"CReM failed for {molecule.smiles}.")
                generated_molecules = []

        case _:
            raise NotImplementedError(f"Unknown masking method '{method}'.")

    return _annotate_whether_proper(generated_molecules)


def compute_fidelity_proxies(
    model: Model,
    molecule: Data,
    context: dict,
    config: dict,
    device: torch.device,
) -> pd.DataFrame:

    model = copy.deepcopy(model)
    model.to(device)
    molecule = copy.deepcopy(molecule)
    molecule.to(device)

    explanation_method_fn = {
        'saliency': partial(saliency_map, model=model),
        'igradients': partial(integrated_gradients, model=model),
        'gradcam': partial(gradcam_node_importances, model=model),
        'gnnexplainer_node_object': partial(gnnexplainer_node_importances, model=model, node_masking='object'),
        'gnnexplainer_node_attributes': partial(gnnexplainer_node_importances, model=model, node_masking='attributes'),
        'random': lambda data: torch.from_numpy(np.random.uniform(-1, +1., size=(data.x.shape[0],)))
    }[context['explanation_method']]

    model.eval()
    with torch.inference_mode():
        print(model)
        prediction_original = model.predict(molecule, binarize=False).item()

    results = [
        context | {
            'masked_atom_percentage': '0%',
            'direction': direction,
            'masking_method': pd.NA,
            'original_molecule_smiles': molecule.smiles,
            'original_molecule_important_atom_indices': pd.NA,
            'masked_molecule_index': -1,
            'masked_molecule_smiles': pd.NA,
            'is_masking_proper': pd.NA,
            'prediction': prediction_original,
        }
        for direction in ('+', '-')
    ]

    for proportion_masked in get_nested(config, 'MASKING', 'atom_proportions'):
        atom_importances = explanation_method_fn(data=molecule).cpu().numpy()
        sorted_indices = atom_importances.argsort()
        k = max(1, int(len(atom_importances)*proportion_masked))
        direction_indices = {
            '+': [int(i) for i in sorted_indices[-k:] if atom_importances[i] > 0],
            '-': [int(i) for i in sorted_indices[:k ] if atom_importances[i] < 0],
        }
        logger.debug(f"{k = } out of {atom_importances = }")

        for masking_method in get_nested(config, 'MASKING', 'methods'):
            for direction, indices in direction_indices.items():
                logger.debug(f"{context=} {proportion_masked=} {direction=} {masking_method=}")
                if not indices:
                    logger.debug('No indices of atoms to mask, skipping.')
                    continue
                masked_molecules = mask_atoms(
                    molecule, indices, masking_method, direction, atom_importances,
                    explanation_method_fn, # context['explanation_method'] == 'random'
                    device
                )
                model.eval()
                with torch.inference_mode():
                    for masked_molecule_index, (masked_molecule, is_change_proper) in enumerate(masked_molecules):
                        masked_molecule.to(device)
                        results.append(context | {
                            'masked_atom_percentage': f"{proportion_masked:.0%}",
                            'direction': direction,
                            'masking_method': masking_method,
                            'original_molecule_smiles': molecule.smiles,
                            'original_molecule_important_atom_indices': indices,
                            'masked_molecule_index': masked_molecule_index,
                            'masked_molecule_smiles': masked_molecule.smiles,
                            'is_masking_proper': is_change_proper,
                            'prediction': model.predict(masked_molecule, binarize=False).item(),
                        })
                    logger.debug(f"explanation_method={context['explanation_method']}  {masking_method=}"
                                 f"  {direction=}  {indices=}  {len(masked_molecules)=}")
    results_df = pd.DataFrame(results)
    return results_df


def process(
    todo: dict,
    config: dict,
    models: Dict[Tuple[str, int], Model],
    datasets: Dict[Tuple[str, int, str], List[Data]],
    output_folder_path: Path,
    device: torch.device,
    dataset_slice: slice = slice(0, -1)
) -> None:

    logger.debug(f"{todo = }")

    model = models[(
        todo['dataset_name'],
        todo['split_index'],
        todo['model_name'],
    )]
    dataset = datasets[(
        todo['dataset_name'],
        todo['split_index'],
    )]

    def _to_positive(mol_index: int) -> int:
        return len(dataset) + mol_index if mol_index < 0 else mol_index

    for molecule_index in range(
        _to_positive(dataset_slice.start),
        _to_positive(dataset_slice.stop)
    ):
        context = todo | {'molecule_index': molecule_index}
        context_values = [
            f"{k.replace('_index', '-') if 'index' in k else ''}{v}"
            for k, v in sorted(context.items())
        ]
        output_file_name = f"{'--'.join(context_values)}.parquet"
        output_file_path = output_folder_path/output_file_name
        if output_file_path.exists():
            logger.info(f"{todo['dataset_name']}/{todo['split_index']}/{molecule_index}: skipped")
            continue

        molecule = dataset[molecule_index]
        df = compute_fidelity_proxies(model, molecule, context, config, device)
        logger.info(f"{todo['dataset_name']}/{todo['split_index']}/{molecule_index}: "
                    f"generated {len(df[ ~df['masking_method'].isnull() ])} molecules")
        df.to_parquet(output_file_path)
        logger.debug(output_file_path)


@click.command()
@click.argument("work_extent_json", type=str)
def main(work_extent_json: str):
    config = load_config()

    work_extent = json.loads(work_extent_json)
    todo_index, todos_reversed, slice_start, slice_stop, log_file_name = \
        work_extent['todo_index'], work_extent['todos_reversed'], \
        work_extent['slice_start'], work_extent['slice_stop'], \
        work_extent['log_file_name']

    logger.add(log_file_name, level='INFO', colorize=False, backtrace=True, diagnose=True)

    models, datasets, todos = load_workspace(config)

    # create output folder if not existent
    output_folder_path = Path.cwd()/get_nested(config, 'output_folder_name')/'masking'
    output_folder_path.mkdir(exist_ok=True)

    # bind arguments know at this point
    process_fn = partial(
        process,
        config=config,
        models=models,
        datasets=datasets,
        output_folder_path=output_folder_path,
        device=select_device(config)
    )

    # load task extents
    dataset_slice = slice(slice_start, slice_stop)

    # batch or single "todo" processing
    if todo_index < 0:
        # loop over all todos
        todo_indices = reversed(range(len(todos))) if todos_reversed else range(len(todos))
        for i in todo_indices:
            set_random_seed(get_nested(config, 'MASKING', 'random_seed') + i)
            logger.info(f"Processing todo #{i}...")
            process_fn(todos[i], dataset_slice=dataset_slice)
            logger.info(f"Finished processing todo #{i}.")
    else:
        # run a single todo
        logger.info(f"Processing todo #{todo_index}...")
        set_random_seed(get_nested(config, 'MASKING', 'random_seed') + todo_index)
        todo = todos[todo_index]
        process_fn(todo, dataset_slice=dataset_slice)
        logger.info(f"Finished processing todo #{todo_index}.")


if __name__ == "__main__":
    main()
