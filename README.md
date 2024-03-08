

fairseq_speechtext project focus on dataset and model part of multi-modual pretraining(i.e: speech and text) for research.
# fairseq_speechtext design principles:

This repository follows the main principle of openness through clarity.

fairseq_speechtext is 

* Support complete recipe.
* Single-file implementation without boilerplate.
* Decoupling of the data and training components.

Avoiding code duplication is not a goal. Readability and hackability are.

--------------------------------------------------------------------------------
# Setup
* base environment requirenment:
  * [PyTorch](http://pytorch.org/) version >= 1.10.0
  * Python version >= 3.7
  * For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
  

# creat python environment
  ```
## python==3.9 pytorch==2.0.1
conda create -n fsq_speechtext python=3.9 -y
 conda activate fsq_speechtext
conda install pytorch==2.0.1  torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/ -y

## python==3.7 pytorch=1.12.1
conda create -n py37 python=3.7 -y
conda activate py37
conda install pytorch==1.12.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge  -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/ -y
or
## for hltsz cluster
## python=3.9 pytorch=2.1.1
conda install pytorch=2.1.1  torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/ -y
or
(recommend in sribd cluster)
pip3 --timeout=1000 install torch==2.1.2 torchaudio  --force-reinstall  --no-cache-dir --index-url https://download.pytorch.org/whl/cu118


## python=3.8 pytorch=1.13.1 cuda11.6
. activate_cuda11.6.sh
conda create -n fsq1131_cuda116 python=3.8 -y
conda activate fsq1131_cuda116
pip install torch==1.13.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

  ```

# compile c++ part and install dependent python package
```
cd /path/to/fsq_speechtext
## Because head node don't contain cuda, the below command can only compile cpp  part into cpu device
pip install -e ./
```

# install flash-attn(optional)
 
```
for example:  cuda11.8 pytorch2.1
#pip install flash-attn --no-build-isolation  ## support flash attention, but it will change before torch version.
We recommend that you can install flash-attn via the below commnad:
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.1/flash_attn-2.5.1+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.5.1+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl


for example: cuda11.6 pytorch1.13.1
We recommend that you can install flash-attn via the below commnad:
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.5/flash_attn-2.3.5+cu116torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install flash_attn-2.3.5+cu116torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
 ```
 ```
* For ASR decoding, you need install the below library:
``` bash
## in order to use fairseq decoder, you now can install it as follows:
## install flashlight text
## it offer beam_search decoder and dictioanry utils
pip install flashlight-text==0.0.4  -i https://pypi.tuna.tsinghua.edu.cn/simple
## install kenlm
## ## offer interget n-gram for ASR decoding
pip install https://github.com/kpu/kenlm/archive/master.zip  ## offer n-gram 
## instll flashlight sequence
export USE_CUDA=1
git clone https://github.com/shanguanma/flashlight_sequence.git && cd flashlight_sequence 
pip install .
```

* For faster training, it will support float16.
 install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
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


# Note:
* `*config`: means that specify setting parameters of model.
* `recipe_shell_script`: means that it is the entry procedure for the proposed method, where you can see the complete recipe for the use of the configuration file, which should all be reproduced or learnt from here. 

# What's New:

* August 2023: [flash_attention2](https://github.com/shanguanma/fairseq_speechtext/blob/main/fairseq/modules/multihead_attention3.py) are supported. check out example [config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/voicelm/voicelm2/config/pretrain/voicelm2_base_librispeech_flash_attention.yaml),[recipe_shell_script](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/voicelm/voicelm2/bash_voicelm2.sh).
* April 2023: interget flashlight library for ASR beam search decoding and kenlm decoding. check out example [viterbi decoding config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/speech_recognition/new/conf/infer_viterbi_librispeech.yaml),[kenlm decoding config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/speech_recognition/new/conf/infer_kenlm_lirispeech.yaml),[fairseq lm decoding config](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/speech_recognition/new/conf/infer_fsqlm_librispeech.yaml);[recipe_shell_script](https://github.com/shanguanma/fairseq_speechtext/blob/main/examples/voicelm/base_voicelm.sh).
