. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
 conda activate fairseq_speechtext

pip uninstall pytorch
conda install pytorch  torchaudio pytorch-cuda=11.7  -c pytorch -c nvidia -y -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/
