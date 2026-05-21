#!/usr/bin/env bash
set -e

# 建议在单独环境里装，避免和你自己的实验环境冲突。
# conda create -n raha_baran python=3.9 -y
# conda activate raha_baran

python -m pip install --upgrade pip setuptools wheel
python -m pip install raha

python - <<'PY'
import raha
print('raha imported OK:', raha.__file__)
PY
