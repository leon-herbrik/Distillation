from dataclasses import dataclass
from pathlib import Path as P
from warnings import warn

import yaml
import torch
from torch.utils.data import Dataset
import h5py


@dataclass
class h5(Dataset):
    target_feature_path: str
    input_feature_path: str
    context_size: int = 16

    def __post_init__(self):
        self.target_files = self._collect_files(self.target_feature_path)
        self.target_index = self._concatenate_index(self.target_files)
        self.target_list = list(self.target_index.keys())
        print(
            f"Loaded target index from {self.target_feature_path} with {len(self.target_index)} keys."
        )

        self.input_files = self._collect_files(self.input_feature_path)
        self.input_index = self._concatenate_index(self.input_files)
        self.input_list = list(self.input_index.keys())
        print(
            f"Loaded input index from {self.input_feature_path} with {len(self.input_index)} keys."
        )

        if not (overlap := self._check_key_overlap(self.input_index,
                                                   self.target_index)):
            warn(
                f"Keys in input and target do not fully overlap. Overlap is {overlap}."
            )
        else:
            print("Keys in input and target fully overlap.")
        self._len = self._calculate_len()

    def __getitem__(self, id, **kwargs):
        if isinstance(id, int):
            id = self.target_list[id]
        # Load input.
        with h5py.File(self.input_index[id], 'r') as data:
            input = torch.from_numpy(data[id][:])
        # Load target.
        with h5py.File(self.target_index[id], 'r') as data:
            target = torch.from_numpy(data[id][:])
        return input, target

    def __len__(self):
        """
        Length depends on the context size, since we extract context_size frames from each video.
        """
        return self._len

    def _calculate_len(self):
        """
        Calculate the length of the dataset.
        """
        length = 0
        for key in self.target_list:
            with h5py.File(self.target_index[key], 'r') as data:
                length += data[key].shape[0]
        return length

    def _collect_files(self, path):
        path = P(path).expanduser()
        return list(path.parent.glob(path.name))

    def _concatenate_index(self, files):
        """Create a concatenated index of h5py datasets (numpy ndarrays) from a list of files."""
        index = {}
        for file in files:
            with h5py.File(file, 'r') as data:
                for key in data.keys():
                    # Set index to filename.
                    if key in index:
                        warn(
                            f"Key {key} already exists in index. Overwriting.")
                    index[key] = file
        return index

    def _check_key_overlap(self, input, target):
        """Check if keys in input and target overlap."""
        set_i = set(input.keys())
        set_t = set(target.keys())
        if set_i == set_t:
            return 1.0
        else:
            # Check amount of keys that are missing in the input.
            return 1 - len(set_i.intersection(set_t)) / len(set_t)

    def _create_integer_based_index(self, index, context_size):
        pass


def test():
    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))
    dataset = h5(**config['dataset'])
    input, target = dataset[0]
    print(f'Shape of input: {input.shape}, shape of target: {target.shape}')
    print(len(dataset))
    pass


if __name__ == "__main__":
    test()
