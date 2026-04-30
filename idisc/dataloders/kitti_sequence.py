import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class KITTISequenceDataset(Dataset):
    CAM_INTRINSIC = {
        "2011_09_26": torch.tensor(
            [
                [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
                [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
            ]
        ),
        "2011_09_28": torch.tensor(
            [
                [7.070493e02, 0.000000e00, 6.040814e02, 4.575831e01],
                [0.000000e00, 7.070493e02, 1.805066e02, -3.454157e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 4.981016e-03],
            ]
        ),
        "2011_09_29": torch.tensor(
            [
                [7.183351e02, 0.000000e00, 6.003891e02, 4.450382e01],
                [0.000000e00, 7.183351e02, 1.815122e02, -5.951107e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.616315e-03],
            ]
        ),
        "2011_09_30": torch.tensor(
            [
                [7.070912e02, 0.000000e00, 6.018873e02, 4.688783e01],
                [0.000000e00, 7.070912e02, 1.831104e02, 1.178601e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 6.203223e-03],
            ]
        ),
        "2011_10_03": torch.tensor(
            [
                [7.188560e02, 0.000000e00, 6.071928e02, 4.538225e01],
                [0.000000e00, 7.188560e02, 1.852157e02, -1.130887e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 3.779761e-03],
            ]
        ),
    }

    HEIGHT = 352
    WIDTH = 1216
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, test_mode, base_path, manifest_path, clip_length=4):
        super().__init__()
        self.base_path = base_path
        self.clip_length = clip_length

        with open(manifest_path) as f:
            manifest = json.load(f)

        split_key = "test_frames" if test_mode else "train_frames"
        self.clips = []
        for drive_key, meta in manifest.items():
            frames = meta.get(split_key, [])
            for i in range(len(frames) - clip_length + 1):
                self.clips.append({
                    "drive_key": drive_key,
                    "date": meta["date"],
                    "image_dir": meta["image_dir"],
                    "frame_indices": frames[i : i + clip_length],
                })

    def preprocess_crop(self, image, intrinsics):
        h_start = int(image.shape[0] - self.HEIGHT)
        w_start = int((image.shape[1] - self.WIDTH) / 2)
        image = image[h_start : h_start + self.HEIGHT, w_start : w_start + self.WIDTH]
        intrinsics = intrinsics.clone()
        intrinsics[0, 2] = intrinsics[0, 2] - w_start
        intrinsics[1, 2] = intrinsics[1, 2] - h_start
        return image, intrinsics

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        raw_intrinsics = self.CAM_INTRINSIC[clip["date"]][:, :3]

        images = []
        clip_intrinsics = None
        for frame_idx in clip["frame_indices"]:
            path = os.path.join(
                self.base_path, clip["image_dir"], f"{frame_idx:010d}.png"
            )
            image = np.asarray(Image.open(path)).astype(np.uint8)
            image, frame_intrinsics = self.preprocess_crop(image, raw_intrinsics)
            if clip_intrinsics is None:
                clip_intrinsics = frame_intrinsics
            image = TF.normalize(TF.to_tensor(image), mean=self.MEAN, std=self.STD)
            images.append(image)

        return {
            "images": torch.stack(images),
            "frame_indices": clip["frame_indices"],
            "sequence_id": clip["drive_key"],
            "camera_intrinsics": clip_intrinsics,
        }
