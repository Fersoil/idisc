#!/usr/bin/env python
"""
Replicate the EXACT finetune.py validation path for E21.
Uses the same code paths as run_finetune() + validate_model_sequential().
"""
import json
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from finetune import validate_model_sequential

BP = "/work/courses/3dv/team17/idisc"
CONFIG_FILE = "configs/kitti/kitti_sam3_video.json"
MODEL_FILE = "/work/courses/3dv/team17/sam3-video-fixed/best_sam_finetuned.pt"

with open(CONFIG_FILE) as f:
    config = json.load(f)

# Match E21 training setup exactly
config["data"]["train_dataset"] = "KITTISequenceDataset"
config["data"]["val_dataset"] = "KITTISequenceDataset"

seed = config["generic"]["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda")

# Build model (same as finetune.py line 301)
print("Building model...", flush=True)
model = IDisc.build(config).to(device)

# E21 used load_pretrained=False (trained from scratch)
# But we need to load the checkpoint. Use the same mechanism as load_pretrained:
print("Loading checkpoint...", flush=True)
ckpt = torch.load(MODEL_FILE, map_location=device)
new_sd = deepcopy({k.replace("module.", ""): v for k, v in ckpt.items()})
missing, unexpected = model.load_state_dict(new_sd, strict=False)
non_pe = [k for k in missing if not k.startswith("pixel_encoder")]
print(f"  Loaded: {len(new_sd)} keys, missing={len(missing)} "
      f"(non-pixel_encoder={len(non_pe)}), unexpected={len(unexpected)}")
if non_pe:
    print(f"  CRITICAL non-pixel_encoder missing: {non_pe[:10]}")
del ckpt, new_sd

# Build validation dataset (same as finetune.py lines 345-361)
data_path = os.path.join(BP, config["data"]["data_root"])
manifest = os.path.join(BP, config["data"].get("manifest_path",
                        "splits/kitti/sequence_manifest.json"))
valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
    test_mode=True,
    base_path=data_path,
    crop=config["data"].get("crop"),
    manifest_path=manifest,
    clip_length=config["data"].get("clip_length", 4),
    stride=config["data"].get("stride"),
)
valid_loader = DataLoader(
    valid_dataset, batch_size=1, shuffle=False,
    num_workers=2, sampler=SequentialSampler(valid_dataset),
    pin_memory=True, drop_last=False,
)
print(f"Val dataset: {len(valid_dataset)} clips", flush=True)

# AMP context (same as finetune.py lines 393-394)
context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

# Run validation (same as finetune.py lines 544-547)
model.eval()
print("Running validation...", flush=True)
with torch.no_grad():
    metrics = validate_model_sequential(
        model, valid_loader, context, device, sam_mode="replace"
    )

print("\n=== Results ===")
for k, v in sorted(metrics.items()):
    print(f"  {k}: {v:.6f}")
