apt update
apt install openmpi-bin openmpi-common libopenmpi-dev
pip install nv
uv sync
source .venv/bin/activate
deepspeed --num_gpus=2 train.py