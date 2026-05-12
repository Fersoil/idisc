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
    min_depth = 0.01
    max_depth = 80

    def __init__(
        self,
        test_mode,
        base_path,
        manifest_path,
        clip_length=4,
        stride=None,
        depth_scale=256,
        crop="eigen",
        sam3_cache_dir=None,
        sam3_top_k=32,
        **kwargs,
    ):
        super().__init__()
        self.base_path = base_path
        self.clip_length = clip_length
        # `stride` controls how far the sliding window advances between
        # successive clips. stride=1 (legacy) makes consecutive clips share
        # clip_length-1 frames; stride=clip_length makes clips disjoint.
        # Disjoint clips reduce duplicate frame reads ~clip_length× and
        # give cleaner per-epoch coverage.
        self.stride = int(stride) if stride is not None else clip_length
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.sam3_cache_dir = sam3_cache_dir
        self.sam3_top_k = sam3_top_k

        with open(manifest_path) as f:
            manifest = json.load(f)

        split_key = "test_frames" if test_mode else "train_frames"
        self.clips = []
        for drive_key, meta in manifest.items():
            frames = meta.get(split_key, [])
            for i in range(0, len(frames) - clip_length + 1, self.stride):
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

    def eval_mask(self, valid_mask):
        if self.test_mode and self.crop is not None:
            h, w = valid_mask.shape[-2:]
            crop_mask = np.zeros_like(valid_mask)
            if "garg" in self.crop:
                crop_mask[
                    int(0.40810811 * h) : int(0.99189189 * h),
                    int(0.03594771 * w) : int(0.96405229 * w),
                ] = 1
            elif "eigen" in self.crop:
                crop_mask[
                    int(0.3324324 * h) : int(0.91351351 * h),
                    int(0.03594771 * w) : int(0.96405229 * w),
                ] = 1
            valid_mask = np.logical_and(valid_mask, crop_mask)
        return valid_mask

    def _sample_train_aug(self):
        """Sample all random decisions once; caller applies them identically to every frame."""
        do_flip = np.random.random() < 0.5
        phot_ops = []
        if np.random.random() < 0.8:
            choice = np.random.randint(5)
            if choice == 0:
                phot_ops = [(TF.adjust_brightness, {"brightness_factor": 10 ** np.random.uniform(-0.2, 0.2)})]
            elif choice == 1:
                phot_ops = [(TF.adjust_contrast, {"contrast_factor": 10 ** np.random.uniform(-0.2, 0.2)})]
            elif choice == 2:
                phot_ops = [(TF.adjust_saturation, {"saturation_factor": 10 ** np.random.uniform(-0.2, 0.2)})]
            elif choice == 3:
                phot_ops = [(TF.adjust_gamma, {"gamma": np.random.uniform(0.8, 1.2)})]
            else:
                phot_ops = [(TF.adjust_hue, {"hue_factor": np.random.uniform(-0.1, 0.1)})]
        return {"do_flip": do_flip, "phot_ops": phot_ops}

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        frame_indices = clip["frame_indices"]
        # KITTI image_dir is "<date>/<drive>/image_02/data"; the annotated
        # depth lives at "<base>/<drive>/proj_depth/groundtruth/image_02/"
        # — drive name only, no date prefix.
        drive_name = clip["image_dir"].split("/")[1]
        depth_dir = os.path.join(drive_name, "proj_depth", "groundtruth", "image_02")

        raws = []
        raw_depths = []
        for fi in frame_indices:
            raws.append(
                np.asarray(
                    Image.open(os.path.join(self.base_path, clip["image_dir"], f"{fi:010d}.png"))
                ).astype(np.uint8)
            )
            depth_path = os.path.join(self.base_path, depth_dir, f"{fi:010d}.png")
            if os.path.isfile(depth_path):
                raw_depths.append(
                    np.asarray(Image.open(depth_path)).astype(np.float32) / self.depth_scale
                )
            else:
                raw_depths.append(None)

        h0 = int(raws[0].shape[0] - self.HEIGHT)
        w0 = int((raws[0].shape[1] - self.WIDTH) / 2)

        clip_intrinsics = self.CAM_INTRINSIC[clip["date"]][:, :3].clone()
        clip_intrinsics[0, 2] -= w0
        clip_intrinsics[1, 2] -= h0

        aug = self._sample_train_aug() if not self.test_mode else {"do_flip": False, "phot_ops": []}

        if aug["do_flip"]:
            clip_intrinsics[0, 2] = self.WIDTH - clip_intrinsics[0, 2]

        images = []
        for raw in raws:
            pil = Image.fromarray(raw[h0 : h0 + self.HEIGHT, w0 : w0 + self.WIDTH])
            if aug["do_flip"]:
                pil = TF.hflip(pil)
            for fn, kw in aug["phot_ops"]:
                pil = fn(pil, **kw)
            images.append(TF.normalize(TF.to_tensor(pil), mean=self.MEAN, std=self.STD))

        depths = []
        masks = []
        for raw_d in raw_depths:
            if raw_d is not None:
                d = raw_d[h0 : h0 + self.HEIGHT, w0 : w0 + self.WIDTH]
                if aug["do_flip"]:
                    d = np.fliplr(d).copy()
                mask = d > self.min_depth
                if self.test_mode:
                    mask = np.logical_and(mask, d < self.max_depth)
                    mask = self.eval_mask(mask)
            else:
                d = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.float32)
                mask = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
            depths.append(torch.from_numpy(d).unsqueeze(0))
            masks.append(torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0))

        sample = {
            "images": torch.stack(images),
            "depths": torch.stack(depths),
            "masks": torch.stack(masks),
            "frame_indices": frame_indices,
            "sequence_id": clip["drive_key"],
            "camera_intrinsics": clip_intrinsics,
        }

        if self.sam3_cache_dir:
            drive_name = clip["image_dir"].split("/")[1]  # e.g. 2011_09_26_drive_0001_sync
            clip_queries = []
            for fi in frame_indices:
                cache_path = os.path.join(self.sam3_cache_dir, drive_name, f"{fi:010d}.pt")
                if os.path.isfile(cache_path):
                    q = torch.load(cache_path, weights_only=True).float()
                    if self.sam3_top_k and q.shape[0] > self.sam3_top_k:
                        _, top_idx = q.norm(dim=-1).topk(self.sam3_top_k)
                        q = q[top_idx]
                    clip_queries.append(q)

            if len(clip_queries) == len(frame_indices):
                sample["sam3_queries"] = torch.stack(clip_queries)  # (T, K, D)
            else:
                sample["sam3_queries"] = torch.zeros(0)
        else:
            sample["sam3_queries"] = torch.zeros(0)

        return sample
