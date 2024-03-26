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

## prepare eval rttm from eval textgrid_dir, it should be orale segement.
`bash scripts/alimeeting/011_prepare_rttm_for_ts_vad_hltsz.sh`


## runing training model with RIRS_NOISZE 
- Training: scripts/alimeeting/020_train_ts_vad_hltsz.sh --stage 0 --stop-stage 0
## Running inferene model on eval dataset
- Inference: bash scripts/alimeeting/030_infer_eval_on_ts_vad_hltsz.sh --stage 0 --stop-stage 0 


## runing training model with RIRS_NOISZE and musan
- Training: scripts/alimeeting/020_train_ts_vad_hltsz.sh --stage 1 --stop-stage 1
## Running inferene model on eval dataset
- Inference: bash scripts/alimeeting/030_infer_eval_on_ts_vad_hltsz.sh --stage 1 --stop-stage 1


eend:
code is modified from https://github.com/Xflick/EEND_PyTorch/tree/master
                      https://github.com/shanguanma/EEND/tree/master
example on mini_librispeech
it is need to install the below package:
yamlargparse
torch
scipy
h5py
numpy

fs_eend:
code is modified from https://github.com/Audio-WestlakeU/FS-EEND/tree/main

example on mini_librispeech
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




