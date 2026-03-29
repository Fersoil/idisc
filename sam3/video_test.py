from sam3.model_builder import build_sam3_video_predictor
import os
from torch.utils.data import DataLoader
from idisc.dataloders.kitti_tracking import KITTITrackingDataset
from idisc.models.idisc import IDisc
import torch.cuda as tcuda
import torch
import json

def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    hidden_states_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    for k, v in predictor.hidden_states.items():
        # print(k, v.shape, end=', ')
        hidden_states_per_frame[k] = v
    return outputs_per_frame, hidden_states_per_frame

def get_hidden_states(video_path):
    video_predictor = build_sam3_video_predictor()
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )

    session_id = response["session_id"]

    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text="blue skirt",
        )
    )

    output, hidden_states = propagate_in_video(video_predictor, session_id)
    return hidden_states

    print("Number of frames:", len(output))

    first_frame = min(output.keys())
    outs = output[first_frame]
    print("First frame index:", first_frame)
    print(list(outs.keys()))
    print("Masks shape:", outs["out_binary_masks"].shape)
    print("Object IDs:", outs["out_obj_ids"])
    print("Scores shape:", outs["out_probs"].shape)
    for frame_idx in sorted(output.keys()):
        outs = output[frame_idx]
        print(
            f"Frame {frame_idx}: "
            f"masks={outs['out_binary_masks'].shape}, "
            f"IDs={outs['out_obj_ids']}, "
            f"scores={outs['out_probs'].shape}"
        )

    print("Part of the hidden state at frame {first_frame}:", hidden_states[first_frame][:2, 0, :3, :3])

root_dir = "/work/courses/3dv/team17/data/kitti_tracking/testing/image_02"
model_file = "/work/courses/3dv/team17/models/kitti_resnet101.pt"
config_file = "/home/lnidogon/idisc/configs/kitti/kitti_r101.json"

with open(config_file, "r") as f:
    config = json.load(f)

device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
model = IDisc.build(config)
model.load_pretrained(model_file)
model = model.to(device)
model.eval()


# PYTHONPATH=$PWD/idisc python idisc/sam3/video_test.py
# source /work/courses/3dv/team17/idisc/.venv/bin/activate
for seq_name in sorted(os.listdir(root_dir)):
    seq_path = os.path.join(root_dir, seq_name)

    if not os.path.isdir(seq_path):
        continue
    hidden_states = None
    print("sequence:", seq_name, "path:", seq_path)
    hidden_states = get_hidden_states(seq_path)
    print(hidden_states[0].shape)
    dataset = KITTITrackingDataset(
        test_mode=True,
        base_path=seq_path,
        crop="eigen",
        benchmark=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(loader):
        print(f"Iteration {i}")
        preds, losses, _ = model(batch["image"].to(device), hidden_states[i], None, None)

