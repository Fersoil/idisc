cd sam3 
python3 -m venv .venv_sam3
source .venv_sam3/bin/activate
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
cd sam3
pip install -e

# This is for faster inference
<!-- # pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
# pip install git+https://github.com/ronghanghu/cc_torch.git -->