import os, random, glob

train_data_dir = os.path.expanduser("~/CIL/data/train")
test_data_dir = os.path.expanduser("~/CIL/data/test")
train_ids = sorted(set(
    f.replace("_rgb.png", "").replace("_depth.npy", "")
    for f in os.listdir(train_data_dir)
    if f.endswith("_rgb.png") or f.endswith("_depth.npy")
))
test_ids = sorted(set(
    f.replace("_rgb.png", "").replace("_depth.npy", "")
    for f in os.listdir(test_data_dir)
    if f.endswith("_rgb.png") or f.endswith("_depth.npy")
))




splits_dir = os.path.expanduser("~/idisc/splits/cil")
os.makedirs(splits_dir, exist_ok=True)

for name, ids, split in [("cil_train.txt", train_ids, "train"), ("cil_test.txt", test_ids, "test")]:
    with open(os.path.join(splits_dir, name), "w") as f:
        for sid in ids:
            if split == "train":
                f.write(f"{split}/{sid}_rgb.png {split}/{sid}_depth.npy\n")
            else:
                f.write(f"{split}/{sid}_rgb.png /home/tkwiecinski/CIL/dummy_depth.npy\n")