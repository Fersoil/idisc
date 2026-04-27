from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "/home/lnidogon/sam3/assets/videos/bedroom.mp4"

response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0,
        text="blue skirt",
    )
)
output = response["outputs"]
print("output keys:", list(output.keys()))
print("Masks:", output["out_binary_masks"].shape)
print("Boxes:", output["out_boxes_xywh"].shape)
print("Object IDs:", output["out_obj_ids"])
print("Scores (probs):", output["out_probs"].shape)