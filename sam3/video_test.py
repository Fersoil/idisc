from sam3.model_builder import build_sam3_video_predictor

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

video_predictor = build_sam3_video_predictor()
video_path = "/home/slazarusic/sam3/assets/videos/bedroom.mp4"

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
