import os, random, glob

data_dir = os.path.expanduser("~/CIL/data/train")
all_ids = sorted(set(
    f.replace("_rgb.png", "").replace("_depth.npy", "")
    for f in os.listdir(data_dir)
    if f.endswith("_rgb.png") or f.endswith("_depth.npy")
))

# 90/10 split
random.seed(42)
random.shuffle(all_ids)
n_val = max(1, int(0.1 * len(all_ids)))
val_ids, train_ids = all_ids[:n_val], all_ids[n_val:]

splits_dir = os.path.expanduser("~/idisc/splits/cil")
os.makedirs(splits_dir, exist_ok=True)

for name, ids in [("cil_train.txt", train_ids), ("cil_val.txt", val_ids)]:
    with open(os.path.join(splits_dir, name), "w") as f:
        for sid in ids:
            f.write(f"train/{sid}_rgb.png train/{sid}_depth.npy\n")