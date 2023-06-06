#!/usr/bin/env bash

. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
 conda activate pt2.0

pip --timeout=1000 install https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230318+cuda11.7.torch2.0.0-cp38-cp38-linux_x86_64.whl
pip3 --timeout=1000  install torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

