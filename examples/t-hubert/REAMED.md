
T-HuBERT:
# Installed:
You can refer to ../README.md

# Data detail:
- speech
   - Librispeech : You can download it from https://www.openslr.org/12
- text
   - librispeech-lm-norm.txt.gz: You can download it from https://www.openslr.org/11/
   - lid.17.bin: You can download it from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin ,it is used to clean corpus for librispeech-lm-norm.txt

- vad: it is used to remove silence segement via this package(https://github.com/zhenghuatan/rVADfast)
- feature extractor(wav2vec_vox_new) of GAN: It is download from https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt

# Source
pretrain T-HuBERT model checkpoint(without finetune), pretrain and finetune data, gan model checkpoint: https://drive.google.com/drive/folders/1HFx-Z43oRKkDIlDwhC7FyptJWFxBleeR?usp=sharing

# Usage
In this recipe, We contains broken down into stages such as:prepared_mfcc_pseudo_label, train_hubert_with_mfcc_label,prepared_hubert_pseudo_label,prepared_text_for_gan,prepared_wav2vec-u_feature_with_sil,prepred_wav2vec-u2_feature_without_sil,train_gan,generate_phncode base_t-hubert_train_ft_infer, numbered in order as 010, 020, etc. These scripts are supposed to be run in order.

# Cation
You need to change input data path, code path, output path in every script dependent on your environment
