import os

import torch


class IFeaturizer:
    def __init__(
            self, y_column="Y", smiles_col="Drug", **kwargs
    ):
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.number_of_features = None
        self.__dict__.update(kwargs)

    def __call__(self, *, split, path):
        path = self.get_path_to_data_dir(path)
        path_to_file = get_path_to_data_file_fn(path)
        print(path_to_file)
        try:
            data_split = {key: torch.load(path_to_file(key), weights_only=False) for key in split.keys()}
        except FileNotFoundError:
            exists = os.path.exists(path)
            if not exists:
                os.makedirs(path)

            self.transform_and_save(split, path_to_file)
            data_split = {key: torch.load(path_to_file(key), weights_only=False) for key in split.keys()}

        try:
            self.number_of_features = data_split["train"][0].x.shape[1]
        except Exception:
            self.number_of_features = 1

        return data_split

    def process(self, df):
        raise NotImplementedError()

    def transform_molecule(self, mol):
        raise NotImplementedError()

    def transform_and_save(self, split, path_to_file):
        for key in split.keys():
            data = self.process(split[key])
            torch.save(data, path_to_file(key))

    def get_path_to_data_dir(self, path):
        return f"{path}/data"

    def get_type(self):
        raise NotImplementedError()


def get_path_to_data_file_fn(path):
    return (
        lambda key: f"{path}/{key}_transformed.pt"
    )


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_hot_encoding(quantity, arr_length=5):
    last_idx = arr_length - 1
    idx = quantity if quantity < last_idx else last_idx
    n_encoding = [0] * arr_length
    n_encoding[idx] = 1
    return n_encoding
