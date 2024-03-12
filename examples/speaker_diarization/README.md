# TS-VAD
## Install
```
git submodule update --init fairseq
pip install --editable fairseq/
```

## Data Prepare
- Download alimeeting Train_Ali_far.tar.gz and Eval_Ali_far.tar.gz from https://www.openslr.org/119/
- Download ecapa-tdnn.model from https://drive.google.com/drive/folders/1AFip2h9W7sCFbzzasL_fAkGUNZOzaTGK
- Download SpeakerEmbedding.zip from https://drive.google.com/file/d/1tNRnF9ouPbPX9jxAh1HkuNBOY9Yx6Pj9/view?usp=sharing
- run preprocess.sh in scripts directory
-  The dataset looks like:

        # alimeeting (Only far data, the first channel is used)
        # ├── Train_Ali
        # │   ├── Train_Ali_far 
        # │     ├── audio_dir
        # │     ├── textgrid_dir
        # ├── Eval_Ali
        # │   ├── Eval_Ali_far 
        # │     ├── audio_dir
        # │     ├── textgrid_dir
        # ├── spk_embed
        # │   ├── SpeakerEmbedding 
        # │     ├── ...

## Running
- Training: scripts/run.sh
- Inference: scripts/run_inf.sh
