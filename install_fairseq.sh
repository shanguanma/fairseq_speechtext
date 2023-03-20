#!/usr/bin/env bash


. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
 conda activate fairseq12.3_py3.9
cd /workspace2/maduo/fairseq
## edit fairseq/pyproject.toml, set "torch==1.11.0"
## edit  fairseq/setup.py set "torch==1.11.0","torchaudio==0.11.0", 
pip install --editable ./   -i https://pypi.tuna.tsinghua.edu.cn/simple
