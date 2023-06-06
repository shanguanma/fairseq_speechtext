#!/usr/bin/env bash


. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
 conda activate fairseq_speechtext
cd /workspace2/maduo/fairseq_speechtext
## edit fairseq/pyproject.toml, set "torch==1.11.0"
## edit  fairseq/setup.py set "torch==1.11.0","torchaudio==0.11.0", 
pip install --editable ./   -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall pytorch
conda install pytorch  torchaudio pytorch-cuda=11.7  -c pytorch -c nvidia -y -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/
pip install soundfile editdistance -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py librosa -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboardX -i  https://pypi.tuna.tsinghua.edu.cn/simple
