#!/usr/bin/env bash
set -e
# 推荐先执行：conda create -n hc37 python=3.7 -y && conda activate hc37
python -m pip install --upgrade "pip<24"
python -m pip install "numpy<1.20" "pandas<1.2" "scipy<1.6" "scikit-learn<0.24" "psycopg2-binary<2.9" "sqlalchemy<1.4"
python -m pip install "torch==1.7.1" -f https://download.pytorch.org/whl/torch_stable.html || true
echo "[INFO] Clone HoloClean if you have not:"
echo "git clone https://github.com/HoloClean/holoclean.git"
echo "cd holoclean && pip install -r requirements.txt"
