apt update
apt install openmpi-bin openmpi-common libopenmpi-dev
pip install nv
uv sync
uv run train.py