# TS-VAD
Here, TS-VAD method is modified by us, We support vary frontend speaker encoder (i.e. ecapa-tdnn, wavlm, cam++,etc)instead of i-vector of origin ts-vad method for speaker embedding part.
The below alimeeting example, We will use ecapa-tdnn as speaker encoder for speaker embedding part, and we use transformer encoder layer instead of blstm layer of origin ts-vad method.

## Data Prepare
- Download alimeeting Train_Ali_far.tar.gz and Eval_Ali_far.tar.gz from https://www.openslr.org/119/

You can use run
```
bash scripts/alimeeting/010_prepare_data_for_ts_vad_hltsz.sh
```
```
The dataset looks like:
        # alimeeting (Only far data, the first channel is used)
        # tree -L 3 /data/alimeeting/
        #/data/alimeeting/
        #├── Eval_Ali
        #│   ├── Eval_Ali_far
	#│   │   ├── audio_dir
	#│   │   ├── Eval.json
	#│   │   ├── target_audio
	#│   │   └── textgrid_dir
	#│   └── Eval_Ali_near
	#│       ├── audio_dir
	#│       └── textgrid_dir
	#└── Train_Ali
	#    └── Train_Ali_far
	#	├── audio_dir
	#	├── target_audio
        #├── textgrid_dir
        #└── Train.json


- Download SpeakerEmbedding.zip from https://drive.google.com/file/d/1tNRnF9ouPbPX9jxAh1HkuNBOY9Yx6Pj9/view?usp=sharing

tree -d -L 2 model_hub/ts_vad/spk_embed/

model_hub/ts_vad/spk_embed/
└── SpeakerEmbedding
    ├── Eval
    └── Train
```
- Download ecapa-tdnn.model from https://drive.google.com/drive/folders/1AFip2h9W7sCFbzzasL_fAkGUNZOzaTGK
note: ecapa-tdnn is trained on voxceleb and finetune on alimeeting dataset.
```
## prepare eval rttm from eval textgrid_dir, it should be orale segement.
bash scripts/alimeeting/011_prepare_rttm_for_ts_vad_hltsz.sh


## runing training model with RIRS_NOISZE
- Training: scripts/alimeeting/020_train_ts_vad_hltsz.sh --stage 0 --stop-stage 0
## Running inferene model on eval dataset
- Inference: bash scripts/alimeeting/030_infer_eval_on_ts_vad_hltsz.sh --stage 0 --stop-stage 0


## runing training model with RIRS_NOISZE and musan
- Training: scripts/alimeeting/020_train_ts_vad_hltsz.sh --stage 1 --stop-stage 1
## Running inferene model on eval dataset
- Inference: bash scripts/alimeeting/030_infer_eval_on_ts_vad_hltsz.sh --stage 1 --stop-stage 1
```
### wavlm+ecapa_tdnn of TS_VAD
# reference: https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
install s3prl:pip install s3prl
download WavLM large(last one of the table) model from https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view

download wavlm pretrain model https://github.com/microsoft/unilm/tree/master/wavlm
WavLM Base+ from https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view
WavLM Larger from https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view




#EEND:
Cation: The working directory is examples/speaker_diarization/, in other words, you run the below script , you need to enter  examples/speaker_diarization/.

```
code is modified from https://github.com/Xflick/EEND_PyTorch/tree/master
                      https://github.com/shanguanma/EEND/tree/master
example on mini_librispeech

you run this script follow number order of name:
1. prepare kaldi format of simulation data based on mini_librispeech
bash scripts/mini_librispeech/010_prepare_mini_librispeech_kaldi_format.sh

2. train eend model on the above simulation data
bash scripts/mini_librispeech/020_train_eend_model.sh

3. infer eend model on dev data of the above simulation data
bash bash scripts/mini_librispeech/030_infer_eend_model.sh


It is need to install the below package:
yamlargparse
torch
scipy
h5py
numpy
```

#FS_EEND(streaming method):
Cation: The working directory is examples/speaker_diarization/, in other words, you run the below script , you need to enter  examples/speaker_diarization/.

```
code is modified from https://github.com/Audio-WestlakeU/FS-EEND/tree/main

example on mini_librispeech
you run this script follow number order of name:
1. prepare kaldi format of simulation data based on mini_librispeech
bash scripts/mini_librispeech/010_prepare_mini_librispeech_kaldi_format.sh

2. train fs_eend model on the above simulation data
bash scripts/mini_librispeech/021_train_fs_eend_model.sh

3. infer fs_eend model on dev data of the above simulation data
bash bash scripts/mini_librispeech/031_infer_fs_eend_model.sh


trainer base on pytorch-lightning
it is need to install the below package:
pytorch-lightning==2.1.2
torch==2.1.1
pyyaml
hyperpyyaml
librosa
soundfile
scipy
torchaudio
torchmetrics
h5py
pyannote.metrics
pyannote.core

```

## clustering_base
pytorch-lightning==2.1.2
torch==2.1.1
pyannote.audio==3.1.1
pip install git+https://github.com/desh2608/spyder.git@main ## for compute DER
kaldi_io ## for xvector
onnxruntime==1.17.1 # for xvector
fastcluster==1.2.4 # for vbx
pip install scikit-learn==1.4.2
then copy the below two file into your python environment, i.e
cp -r clustering_based/spectral/file/_spectral.py /home/maduo/.conda/envs/fsq_sptt/lib/python3.9/site-packages/sklearn/cluster/
cp -r clustering_based/spectral/file/_spectral_embedding.py /home/maduo/.conda/envs/fsq_sptt/lib/python3.9/site-packages/sklearn/manifold/

so, sklearn will support process diarization overlap case in spectral cluster method.
you can run the below code:
scripts/alimeeting/clustering_based/051_spectral_ovl.sh
scripts/alimeeting/clustering_based/051_spectral.sh

# for clustering_based/cder
pip install pyannote.core
