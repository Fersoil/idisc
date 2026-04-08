import os
import numpy as np
import torch
from PIL import Image
from .dataset import BaseDataset


class CILDataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor([
            [518.8579, 0.0,      325.5824],
            [0.0,      519.4696, 253.7362],
            [0.0,      0.0,      1.0     ],
        ])
    }
    min_depth = 0.01
    max_depth = 80.0        # adjust if your dataset is indoor (use 10.0)
    train_split = "/home/tkwiecinski/idisc/splits/cil/cil_train.txt"
    val_split  = "/home/tkwiecinski/idisc/splits/cil/cil_val.txt"
    test_split = "/home/tkwiecinski/idisc/splits/cil/cil_test.txt"

    
    def __init__(self, test_mode, base_path, depth_scale=1.0,
                 crop=None, benchmark=False, augmentations_db={},
                 masked=True, normalize=True, **kwargs):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.depth_scale = depth_scale
        self.crop = crop
        self.height = 560   # it seems that all the images are square
        self.width = 560    
        self.masked = masked
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join(self.split_file)) as f:
            for line in f:
                parts = line.strip().split(" ")
                img_info = {
                    "image_filename": os.path.join(self.base_path, parts[0]),
                    "annotation_filename_depth": os.path.join(self.base_path, parts[1]),
                }
                self.dataset.append(img_info)
        print(f"Loaded {len(self.dataset)} samples.")

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.dataset[idx]["image_filename"]).convert("RGB"))
        if not self.test_mode:
            depth = np.load(self.dataset[idx]["annotation_filename_depth"]).astype(np.float32)
        else:
            depth = np.zeros((self.height, self.width), dtype=np.float32)
        depth = depth / self.depth_scale
        info = self.dataset[idx].copy()
        info["camera_intrinsics"] = self.CAM_INTRINSIC["ALL"].clone()
        image, gts, info = self.transform(image=image, gts={"depth": depth}, info=info)
        return {"image": image, "gt": gts["gt"], "mask": gts["mask"]}

    def get_pointcloud_mask(self, shape):
        mask = np.ones(shape)   # no border crop for CIL data
        return mask

    def preprocess_crop(self, image, gts=None, info=None):
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            mask = (depth > self.min_depth).astype(np.uint8)
            if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth).astype(np.uint8)
            new_gts["gt"] = depth
            new_gts["mask"] = mask
        return image, new_gts, info