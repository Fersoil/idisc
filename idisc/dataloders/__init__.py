from .argoverse import ArgoverseDataset
from .dataset import BaseDataset
from .ddad import DDADDataset
from .diode import DiodeDataset
from .kitti import KITTIDataset
from .kitti_sequence import KITTISequenceDataset
from .nyu import NYUDataset
from .nyu_normals import NYUNormalsDataset
from .sunrgbd import SUNRGBDDataset


def select_dataset_cls(dataset_name, dataset_mode):
    """Pick the dataset class name from the resolved config. Image mode picks by
    dataset (kitti -> KITTIDataset, nyu -> NYUDataset); video mode is KITTI-only
    (clip sampling). Keeps the SAM3 train/eval paths dataset-agnostic instead of
    hardcoding KITTI."""
    if dataset_mode == "video":
        return "KITTISequenceDataset"
    return {"nyu": "NYUDataset"}.get(dataset_name, "KITTIDataset")


__all__ = [
    "BaseDataset",
    "NYUDataset",
    "NYUNormalsDataset",
    "KITTIDataset",
    "KITTISequenceDataset",
    "ArgoverseDataset",
    "DDADDataset",
    "DiodeDataset",
    "SUNRGBDDataset",
    "select_dataset_cls",
]
