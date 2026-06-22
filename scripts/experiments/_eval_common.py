import os

from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc


def build_eval_model(cfg, checkpoint_path, device):
    model = IDisc.build(cfg).to(device)
    model.load_pretrained(checkpoint_path)
    model.eval()
    return model


def _build_dataset(cfg, test_mode, augmentations=None):
    dataset_mode = cfg["run"]["dataset_mode"]
    dataset_cls = custom_dataset.select_dataset_cls(cfg["dataset_name"], dataset_mode)
    data = cfg["data"]
    kwargs = dict(
        test_mode=test_mode,
        base_path=os.path.join(cfg["paths"]["base_path"], data["data_root"]),
        crop=data["crop"],
    )
    if augmentations is not None:
        kwargs["augmentations_db"] = augmentations
    if dataset_mode == "video":
        kwargs["manifest_path"] = data["manifest_path"]
        kwargs["clip_length"] = data["clip_length"]
    return getattr(custom_dataset, dataset_cls)(**kwargs)


def build_val_loader(cfg):
    dataset = _build_dataset(cfg, test_mode=True)
    return DataLoader(
        dataset, batch_size=1, num_workers=2,
        sampler=SequentialSampler(dataset), pin_memory=True,
    )


def build_train_loader(cfg, batch_size):
    dataset = _build_dataset(cfg, test_mode=False, augmentations=cfg["data"]["augmentations"])
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
