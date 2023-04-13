
fairseq_speechtext project focus on joint speech and text multi-modual pretraining

--------------------------------------------------------------------------------
# Requirements and Installation


* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
## for pytorch==2.0  version
. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
conda create -n fairseq_speechtext python=3.9 -y
 conda activate fairseq_speechtext
cd /workspace2/maduo/fairseq_speechtext
pip install --editable ./   -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall pytorch
conda install pytorch  torchaudio pytorch-cuda=11.7  -c pytorch -c nvidia -y -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/
pip install soundfile editdistance  tensorboardX   -i https://pypi.tuna.tsinghua.edu.cn/simple



## for pytorch 1.* version(e.g. pytorch=1.11.0)
. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
conda create -n fairseq_speechtext python=3.9 -y
 conda activate fairseq_speechtext
cd /workspace2/maduo/fairseq_speechtext
## edit fairseq/pyproject.toml, set "torch==1.11.0"
## edit  fairseq/setup.py set "torch==1.11.0","torchaudio==0.11.0",
pip install --editable ./   -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install soundfile editdistance tensorboardX  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install librosa h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install hydra-core --upgrade
pip3 install torch==1.11.0+cu113  torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install bitarray tqdm  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade --force-reinstall sacrebleu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install  timm  torchvision==0.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

