from dataclasses import dataclass
from pathlib import Path as P
from warnings import warn
import json

import yaml
import torch
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm


@dataclass
class h5Dataset(Dataset):
    target_feature_path: str
    input_feature_path: str
    split_index_path: str
    split: str
    context_size: int = 16
    target_context_size: int = 1

    def __post_init__(self):
        self.split_index = json.load(
            open(P(self.split_index_path).expanduser()))[self.split]
        self.target_files = self._collect_files(self.target_feature_path)
        self.target_index = self._concatenate_index(self.target_files,
                                                    self.split_index)
        self.target_list = list(self.target_index.keys())
        print(
            f"Loaded target index from {self.target_feature_path} with {len(self.target_index)} keys."
        )
        self.target_int_index = self._create_integer_based_index(
            self.target_index, self.target_context_size, 'target_int_index')

        self.input_files = self._collect_files(self.input_feature_path)
        self.input_index = self._concatenate_index(self.input_files,
                                                   self.split_index)
        self.input_list = list(self.input_index.keys())
        print(
            f"Loaded input index from {self.input_feature_path} with {len(self.input_index)} keys."
        )
        self.input_int_index = self._create_integer_based_index(
            self.input_index, self.context_size, 'input_int_index')

        if len(self.input_int_index) != len(self.target_int_index):
            raise ValueError(
                "Input and target index have different lengths. This should not happen."
            )
        else:
            print(
                f"Input and target index have the same length: {len(self.input_int_index)}"
            )

        if not (overlap := self._check_key_overlap(self.input_index,
                                                   self.target_index)):
            warn(
                f"Keys in input and target do not fully overlap. Overlap is {overlap}."
            )
        else:
            print("Keys in input and target fully overlap.")

    def __getitem__(self, id, **kwargs):
        if isinstance(id, int):
            input_slice, target_slice = self.input_int_index[
                id], self.target_int_index[id]
            return self._load_data_from_index(input_slice, target_slice)
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
        return len(self.target_int_index)

    def _load_data_from_index(self, input_slice, target_slice):
        """
        Load data from the index. The index is a list of tuples, where each tuple contains
        the video id, the start and the end frame of a batch of frames.
        """
        input_id, input_start, input_end = input_slice
        target_id, target_start, target_end = target_slice
        with h5py.File(self.input_index[input_id], 'r') as data:
            input = torch.from_numpy(data[input_id][input_start:input_end])
        with h5py.File(self.target_index[target_id], 'r') as data:
            target = torch.from_numpy(data[target_id][target_start:target_end])
        # Remove leading singleton dimensions.
        input, target = input.squeeze(0), target.squeeze(0)
        return input, target

    def _collect_files(self, path):
        path = P(path).expanduser()
        return list(path.parent.glob(path.name))

    def _concatenate_index(self, files, split_ids=None):
        """Create a concatenated index of h5py datasets (numpy ndarrays) from a list of files."""
        index = {}
        # Check if any split_id is missing in the files.
        for file in files:
            with h5py.File(file, 'r') as data:
                for key in data.keys():
                    if split_ids and key not in split_ids:
                        continue
                    # Set index to filename.
                    if key in index:
                        warn(
                            f"Key {key} already exists in index. Overwriting.")
                    index[key] = file
                pass
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

    def _create_integer_based_index(self, index, context_size, name):
        """
        Go through all videos in the index, extract batches of context_size frames.
        Discard the last batches that are smaller than context_size.
        For each batch, store a tuple of the video id and the start and end frame,
        so that we can later retrieve the data based on an integer index.
        """
        # Check if the index already exits and load it.
        # If it does not exist, create it and store it.
        if not (path := P(f"{name}.json")).exists():
            # Create the file.
            json.dump({}, open(path, 'w'))
        if not self.split in (data := json.load(open(path))):
            int_index = []
            for id, file_name in tqdm(index.items(),
                                      total=len(index),
                                      desc="Creating integer index"):
                with h5py.File(file_name, 'r') as h5file:
                    video = h5file[id][:]
                for i in range(0, len(video), context_size):
                    if i + context_size > len(video):
                        break
                    int_index.append((id, i, i + context_size))
            data[self.split] = int_index
        json.dump(data, open(path, 'w'))
        return data[self.split]


def test():
    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))
    dataset = h5Dataset(**config['dataset'], split='test')
    input, target = dataset[0]
    print(f'Shape of input: {input.shape}, shape of target: {target.shape}')
    print(f"Length: {len(dataset)}")
    pass


if __name__ == "__main__":
    test()
