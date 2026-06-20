import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc


def build_eval_model(cfg, checkpoint_path, device):
    model = IDisc.build(cfg).to(device)
    model.load_pretrained(checkpoint_path)
    model.eval()
    return model


def build_val_loader(cfg):
    dataset_mode = cfg["run"]["dataset_mode"]
    dataset_cls = custom_dataset.select_dataset_cls(cfg["dataset_name"], dataset_mode)
    data = cfg["data"]
    data_path = os.path.join(cfg["paths"]["base_path"], data["data_root"])
    if dataset_mode == "video":
        dataset = getattr(custom_dataset, dataset_cls)(
            test_mode=True, base_path=data_path, crop=data["crop"],
            manifest_path=data["manifest_path"], clip_length=data["clip_length"],
        )
    else:
        dataset = getattr(custom_dataset, dataset_cls)(
            test_mode=True, base_path=data_path, crop=data["crop"],
        )
    return DataLoader(
        dataset, batch_size=1, num_workers=2,
        sampler=SequentialSampler(dataset), pin_memory=True,
    )
