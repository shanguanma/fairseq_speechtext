

fairseq_speechtext project focus on multi-modual pretraining(i.e: speech and text) for research.
Fairseq_speechtext design principles:

* simple: support complete recipe.
          Single-file implementation without boilerplate.
          Decoupling of the data and training components.

Avoiding code duplication is not a goal. Readability and hackability are.

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
pip install  timm  torchvision==0.15.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

for fixing ImportError: cannot import name 'get_ref_type' from 'omegaconf._utils'
you  should pip install omegaconf==2.1.2 
##for fairseq c++ part compile
python setup.py build_ext --inplace




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

for fixing ImportError: cannot import name 'get_ref_type' from 'omegaconf._utils'
you  should pip install omegaconf==2.1.2

##for fairseq c++ part compile
python setup.py build_ext --inplace
```

# For ASR decoding, you need install the below library:
```
## in order to use fairseq decoder, you now can install it as follows:
## install flashlight text
## it offer beam_search decoder and dictioanry utils
pip install flashlight-text  -i https://pypi.tuna.tsinghua.edu.cn/simple
## install kenlm
## ## offer interget n-gram for ASR decoding
pip install https://github.com/kpu/kenlm/archive/master.zip  ## offer n-gram 
## instll flashlight sequence
export USE_CUDA=1
git clone https://github.com/flashlight/sequence && cd sequence 
pip install .
```

# For faster training, it will support float16.
 install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .


# Note:
* `*config`: means that specify setting parameters of model.
* `recipe_shell_script`: means that it is the entry procedure for the proposed method, where you can see the complete recipe for the use of the configuration file, which should all be reproduced or learnt from here. 

# What's New:

* August 2023: [flash_attention2](https://github.com/shanguanma/fairseq_speechtext/blob/main/fairseq/modules/multihead_attention3.py) are supported. check out example [config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/voicelm/voicelm2/config/pretrain/voicelm2_base_librispeech_flash_attention.yaml),[recipe_shell_script](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/voicelm/voicelm2/bash_voicelm2.sh).
* April 2023: interget flashlight library for ASR beam search decoding and kenlm decoding. check out example [viterbi decoding config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/speech_recognition/new/conf/infer_viterbi_librispeech.yaml),[kenlm decoding config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/speech_recognition/new/conf/infer_kenlm_lirispeech.yaml),[fairseq lm decoding config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/speech_recognition/new/conf/infer_fsqlm_librispeech.yaml);[recipe_shell_script](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/voicelm/base_voicelm.sh).
