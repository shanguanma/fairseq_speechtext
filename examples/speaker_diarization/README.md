# TS-VAD

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
