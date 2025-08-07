#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from typing import Literal, Dict

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from tdc.single_pred import ADME, Tox
from loguru import logger

from common import (
    load_config,
    get_nested,
    select_device,
    set_random_seed
)
from models import get_model_class
from featurizer import Featurizer
from transform import Transform
from trainer import Trainer


def setup_models(
    model_variants_config: dict,
    featurizer: Featurizer,
    task_type: Literal['regression', 'classification'],
) -> Dict[str, torch.nn.Module]:
    """
    Initializes model variants as defined in the configuration.

    Returns:
        Dict[str, Model]: A dictionary of initialized models keyed by variant name.
    """
    models: Dict[str, torch.nn.Module] = {}

    for variant_name, variant_config in model_variants_config.items():
        # setup a concrete model object
        model_class = get_model_class(variant_config['architecture'])
        kwargs = {'task_type': task_type} | variant_config | featurizer.feature_counts_dict()
        model = model_class(**kwargs)

        # add to the collection
        if variant_name in models:
            raise KeyError(f"Duplicate model variant name detected: '{variant_name}'.")
        models[variant_name] = model
        logger.debug(f"{variant_name} for {task_type}, params: {model.num_trainable_parameters()}")

    return models


def train_models(config: dict) -> None:

    def from_config(*args, **kwargs):
        return get_nested(config, *args, **kwargs)

    device = select_device(config)
    featurizer = Featurizer()
    transformer = Transform(
        featurizer=featurizer,
        min_heavy_atoms=from_config('DATA', 'TRANSFORM', 'min_heavy_atoms'),
        protonate=from_config('DATA', 'TRANSFORM', 'protonate'),
    )
    output_folder_path = Path.cwd()/from_config('output_folder_name')

    for dataset_name, dataset_properties in (
        from_config('DATA', 'ADME_DATASETS')
      | from_config('DATA', 'TOX_DATASETS')
    ).items():

        logger.info(f"DATASET: {dataset_name}")
        data = None
        if dataset_name in from_config('DATA', 'ADME_DATASETS'):
            data = ADME(dataset_name, path=output_folder_path)
        elif dataset_name in from_config('DATA', 'TOX_DATASETS'):
            data = Tox(dataset_name, path=output_folder_path)
        assert data
        logger.info(f"[dataset {dataset_name}] Molecule count: {len(data)}.")

        # initialize models
        task_type = dataset_properties['task_type']
        model_variants_config = get_nested(config, 'MODEL', 'VARIANTS')

        # generate data splits
        for split_index in range(n_splits := from_config('DATA', 'SPLIT', 'count')):

            # make a data split (partitioning) and transform (clean & featurize) each partition
            logger.info(f"[dataset {dataset_name}, split {split_index + 1}/{n_splits}] Transforming dataset...")
            split_seed = from_config('DATA', 'SPLIT', 'random_seed') + split_index
            set_random_seed(split_seed)
            split_fractions = from_config('DATA', 'SPLIT', 'FRACTION')
            split_data = data.get_split(
                method='random',
                seed=split_seed,
                frac=[
                    split_fractions.get('train', 0.7),
                    split_fractions.get('valid', 0.1),
                    split_fractions.get('test',  0.2),
                ]
            )
            transformed_split_data = transformer(split_data)

            # dump test set to a file
            test_dataset_file_name = f"dataset_{dataset_name}--split_{split_index}--partition_test.pth"
            test_dataset_file_path = output_folder_path / test_dataset_file_name
            torch.save(transformed_split_data['test'], test_dataset_file_path)
            logger.info(f"[dataset {dataset_name}, split {split_index + 1}/{n_splits}] "
                        f"Saved partition 'test' to a file.")

            # train models for the current data split
            models = setup_models(model_variants_config, featurizer, task_type)
            for model_index, (model_name, model) in enumerate(models.items()):

                training_config = from_config('MODEL', 'TRAINING')
                model_file_name = f"dataset_{dataset_name}--split_{split_index}--model_{model_name}.pth"
                model_file_path = output_folder_path / model_file_name

                # has the model been already trained? do we want to force re-training?
                if model_file_path.exists() and not training_config['even_if_model_file_exists']:
                    logger.warning(f"[dataset {dataset_name}] Note: Found model file '{model_file_path.name}'"
                                   ' => training skipped.')
                    continue

                # model training
                logger.info(f"[dataset {dataset_name}, split {split_index + 1}/{n_splits}, "
                            f"model {model_index + 1}/{len(models)}] Training {model_name}...")
                training_seed = training_config['random_seed'] + model_index
                set_random_seed(training_seed)
                dataloaders = {
                    partition_name: GraphDataLoader(
                        partition_dataset,
                        shuffle=partition_name == 'train',
                        **training_config['LOADER'],
                    )
                    for partition_name, partition_dataset in transformed_split_data.items()
                }
                trainer = Trainer(model, model_file_path, dataloaders, device, **training_config['TRAINER'])
                trainer.fit()
                assert model_file_path.exists(), f"Non-existent model file '{model_file_path}'."
                logger.info(f"[dataset {dataset_name}, split {split_index + 1}/{n_splits}, "
                            f"model {model_index + 1}/{len(models)}] Saved {model_name} to a file.")


if __name__ == '__main__':
    conf = load_config()
    train_models(conf)
