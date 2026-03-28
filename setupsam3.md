module add cuda/12.8


cd ~/idisc/sam3 
python3 -m venv /work/scratch/lnidogon/.venv_sam3
source /work/scratch/lnidogon/.venv_sam3/bin/activate
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
cd ~/idisc/sam3
pip install -e ".[notebooks]"

hf auth login

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/sam3', local_dir='/work/scratch/lnidogon/sam3_checkpoints')
"

pip install -e ".[notebooks]"
# This is for faster inference
<!-- # pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
# pip install git+https://github.com/ronghanghu/cc_torch.git -->