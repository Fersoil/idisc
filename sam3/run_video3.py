from transformers import Sam3Model, Sam3Processor
from transformers.video_utils import load_video
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

config = Sam3Model.config_class.from_pretrained("facebook/sam3")
config.output_hidden_states = True

model = Sam3Model.from_pretrained("facebook/sam3", config=config).to(device, dtype=torch.bfloat16)
processor = Sam3Processor.from_pretrained("facebook/sam3")

video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

for frame_idx, frame in enumerate(video_frames[:10]):
    inputs = processor(
        images=frame,
        text="person",
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    last_hidden = hidden_states[-1]

    print(f"Frame {frame_idx}: {last_hidden.shape}")