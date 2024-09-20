from pathlib import Path as P

import h5py

target_feature_path = P(
    '~/GroundVQA/data/unified/egovlp_internvideo_subsampled.hdf5').expanduser(
    )
input_feature_path = P(
    '~/vqgancodes/vqgan_codes_nlq_train_val_test.hdf5').expanduser()

# Load both files and see if all keys are identical.
with h5py.File(target_feature_path, 'r') as target_file:
    target_keys = set(target_file.keys())
with h5py.File(input_feature_path, 'r') as input_file:
    input_keys = set(input_file.keys())

print(f"Target keys: {len(target_keys)}")
print(f"Input keys: {len(input_keys)}")
print(f"Overlap: {len(target_keys.intersection(input_keys))}")
print(f"Target = Input? {target_keys == input_keys}")
