import sys
import random

import json
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L

from dataset.dataset import h5Dataset
from util.util import are_all_A100
from model.lightning_module import LightningModule


def convert(trainer, model, inference_dataloader):
    predictions = trainer.predict(model, inference_dataloader)
    print(predictions[0])
    dict_predictions = batches_to_dict(predictions)
    # Store dict.
    # Add random prefix to file name due to multiprocessing.
    random_prefix = str(random.randint(0, 100000))
    with open(f'predictions_{random_prefix}.json', 'w') as f:
        json.dump(dict_predictions, f)


def batches_to_dict(batches):
    elems_by_key = {}
    for batch in batches:
        x, (key, start, end) = batch
        elems_by_key[key] = (x, start, end)
    return elems_by_key


if __name__ == "__main__":

    # Set matmul precision if all GPUs are A100.
    if are_all_A100():
        torch.set_float32_matmul_precision("medium")

    config_path = sys.argv[1] if len(
        sys.argv) > 1 else "config_bender_4_frames_inference.yaml"

    checkpoint_path = sys.argv[2] if len(
        sys.argv
    ) > 2 else './checkpoints/GT-D12-F4-CB2-CA2-epoch=20-val_loss=0.52.ckpt'

    config = yaml.safe_load(open(config_path))

    model = LightningModule.load_from_checkpoint(checkpoint_path,
                                                 config=config['model'])
    vocabulary = model.model.vocabulary
    # Create collate function arguments.
    collate_fn_args = {
        'mask_token': vocabulary['[mask]'],
        'codebook_size': config['model']['codebook_size'],
        'past_shift': vocabulary.past_shift,
        'future_shift': vocabulary.future_shift,
        'feature_token': vocabulary['[feat]'],
        'sep_token': vocabulary['[sep]'],
    }
    inference_dataset = h5Dataset(
        **config['dataset'],
        split='train',
        **collate_fn_args,
    )

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=config['trainer']['batch_size'],
        shuffle=False,
        num_workers=128,
        collate_fn=inference_dataset.collate_fn)

    trainer = L.Trainer(limit_predict_batches=100, devices=1)

    convert(trainer, model, inference_dataloader)
