from idisc.dataloders.kitti_sequence import KITTISequenceDataset

ds = KITTISequenceDataset(
    test_mode=True,
    base_path="/path/to/kitti",
    manifest_path="splits/kitti/sequence_manifest.json",
    clip_length=4,
)

print(f"Total clips: {len(ds)}")

sample = ds[0]
print(sample["sequence_id"])           # drive key string
print(sample["frame_indices"])         # 4 consecutive ints
print(sample["images"].shape)          # torch.Size([4, 3, 352, 1216])
print(sample["images"].min(), sample["images"].max())  # roughly [-2, 2]

# Check sliding window: adjacent clips share T-1 frames if same drive
sample2 = ds[1]
print(sample2["frame_indices"])

if sample["sequence_id"] == sample2["sequence_id"]:
    overlap = [f for f in sample["frame_indices"] if f in sample2["frame_indices"]]
    assert len(overlap) == 3, f"Expected 3 overlapping frames, got {len(overlap)}: {overlap}"
    print("Sliding window overlap OK")
else:
    print("Drive boundary between clip 0 and 1 — no overlap expected")
