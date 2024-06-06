#!/usr/bin/env bash
stage=0
stop_stage=100
# VAD Hyperparameters (tuned on eval)
onset=0.6
offset=0.5
min_duration_on=0.4
min_duration_off=0.15

. ./cmd.sh
. ./path_for_fsq_sptt.sh
. ./utils/parse_options.sh

DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base
mkdir -p exp

model_dir=/home/maduo/model_hub/vad/pyannote_segmentation/
sctk_dir=SCTK-2.4.12/src/md-eval/

## step 1: download pyannote segmentation model
## step 2: resegmentation on target dataset (i.e. alimeeting) , called as segmentation finetune
## step 3: get target dataset vad using the above segmentation model.

export PATH=/home/maduo/installed/sox-14.4.2/bin:$PATH # /usr/bin/soxi
if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then

  if [ -f $model_dir/alimeeting_epoch0_step2600.ckpt ]; then
    echo "Found existing AliMeeting pyannote segmentation model, skipping training..."
  else
    mkdir -p exp/pyannote/alimeeting/lists
    cp $DATA_DIR/{train,eval}/rttm/* exp/pyannote/alimeeting/lists/
    for f in $DATA_DIR/{train,eval}/audios/*; do
      filename=$(basename $f .wav)
      duration=$(soxi -D $f)
      echo "$filename 1 0.00 $duration" > exp/pyannote/alimeeting/lists/${filename}.uem
    done
    ls -1 $DATA_DIR/train/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/alimeeting/lists/train.meetings.txt
    ls -1 $DATA_DIR/eval/audios/*.wav | xargs -n 1 basename | sed 's/\.[^.]*$//' > exp/pyannote/alimeeting/lists/eval.meetings.txt
    echo "Fine tuning pyannote segmentation model on AliMeeting..."
    export PYANNOTE_DATABASE_CONFIG=clustering_based/pyannote/database.yml
    python clustering_based/pyannote/train_seg_finetune.py \
	    --dataset AliMeeting \
	    --exp_dir exp/pyannote/alimeeting\
	    --pyannote_segmentation_model $model_dir/pytorch_model.bin
    ## logging
    # Fine tuning pyannote segmentation model on AliMeeting...
#'AMI.SpeakerDiarization.only_words' found in /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/clustering_based/pyannote/database.yml does not define the 'scope' of speaker labels (file, database, or global). Setting it to 'file'.
#'AISHELL-4.SpeakerDiarization.only_words' found in /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/clustering_based/pyannote/database.yml does not define the 'scope' of speaker labels (file, database, or global). Setting it to 'file'.
#'AliMeeting.SpeakerDiarization.only_words' found in /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/clustering_based/pyannote/database.yml does not define the 'scope' of speaker labels (file, database, or global). Setting it to 'file'.
#/home/maduo/.conda/envs/fsq_sptt/lib/python3.9/site-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
#  torchaudio.set_audio_backend("soundfile")
#'AMI.SpeakerDiarization.only_words' found in /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/clustering_based/pyannote/database.yml does not define the 'scope' of speaker labels (file, database, or global). Setting it to 'file'.
#'AISHELL-4.SpeakerDiarization.only_words' found in /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/clustering_based/pyannote/database.yml does not define the 'scope' of speaker labels (file, database, or global). Setting it to 'file'.
#'AliMeeting.SpeakerDiarization.only_words' found in /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/clustering_based/pyannote/database.yml does not define the 'scope' of speaker labels (file, database, or global). Setting it to 'file'.
#Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.1.2. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../../../../model_hub/vad/pyannote_segmentation/pytorch_model.bin`
#Model was trained with pyannote.audio 0.0.1, yours is 3.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
#Model was trained with torch 1.10.0+cu102, yours is 2.1.1+cu118. Bad things might happen unless you revert torch to 1.x.
#Protocol AliMeeting.SpeakerDiarization.only_words does not precompute the output of torchaudio.info(): adding a 'torchaudio.info' preprocessor for you to speed up dataloaders. See pyannote.database documentation on how to do that yourself.
#GPU available: False, used: False
#TPU available: False, using: 0 TPU cores
#IPU available: False, using: 0 IPUs
#HPU available: False, using: 0 HPUs
#
#  | Name              | Type             | Params | In sizes      | Out sizes
#---------------------------------------------------------------------------------------------------------------------
#0 | sincnet           | SincNet          | 42.6 K | [1, 1, 80000] | [1, 60, 293]
#1 | lstm              | LSTM             | 1.4 M  | [1, 293, 60]  | [[1, 293, 256], [[8, 1, 128], [8, 1, 128]]]
#2 | linear            | ModuleList       | 49.4 K | ?             | ?
#3 | classifier        | Linear           | 516    | [1, 293, 128] | [1, 293, 4]
#4 | activation        | Sigmoid          | 0      | [1, 293, 4]   | [1, 293, 4]
#5 | validation_metric | MetricCollection | 0      | ?             | ?
#---------------------------------------------------------------------------------------------------------------------
#1.5 M     Trainable params
#0         Non-trainable params
#1.5 M     Total params
#5.892     Total estimated model params size (MB)
#Epoch 0: 100%|█| 2600/2600 [1:06:12<00:00,  0.65it/s, v_num=3409, DiarizationErrorRate=0.199, DiarizationErrorRate/Confusion=0.0346, DiarizationErrorRate/FalseAlarm=0.0
#`Trainer.fit` stopped: `max_epochs=1` reached.
#Epoch 0: 100%|█| 2600/2600 [1:06:12<00:00,  0.65it/s, v_num=3409, DiarizationErrorRate=0.199, DiarizationErrorRate/Confusion=0.0346, DiarizationErrorRate/FalseAlarm=0.0

   #cp exp/pyannote/alimeeting/lightning_logs/version_0/checkpoints/epoch=0-step=2492.ckpt $model_dir/alimeeting_epoch0_step2492.ckpt
  fi
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  cp exp/pyannote/alimeeting/lightning_logs/version_3409/checkpoints/epoch\=0-step\=2600.ckpt $model_dir/alimeeting_epoch0_step2600.ckpt
  for part in eval test; do
    echo "Running pyannote VAD on ${part}..."
    (
    for audio in $(ls $DATA_DIR/$part/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt

      $train_cmd $EXP_DIR/${part}/log/vad/vad_${filename}.log \
        python clustering_based/vad/pyannote_vad.py \
          --model $model_dir/alimeeting_epoch0_step2600.ckpt \
          --in-dir $DATA_DIR/$part/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/$part/vad \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} &

    done
    wait
    )
    rm exp/list_*
  done
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Onset: $onset Offset: $offset Min-duration-on: $min_duration_on Min-duration-off: $min_duration_off"
  for part in eval test; do
    echo "Evaluating ${part} VAD output"
    cat $DATA_DIR/${part}/rttm/* > $EXP_DIR/ref_${part}.rttm
    [ -f $EXP_DIR/hyp_${part}_vad.rttm ] && rm $EXP_DIR/hyp_${part}_vad.rttm
    for x in $EXP_DIR/${part}/vad/*; do
      session=$(basename $x .lab)
      awk -v SESSION=${session} \
        '{print "SPEAKER", SESSION, "1", $1, $2-$1, "<NA> <NA> sp <NA> <NA>"}' $x >> $EXP_DIR/hyp_${part}_vad.rttm
    done
    #$sctk_dir/md-eval.pl -r $EXP_DIR/ref_${part}.rttm -s $EXP_DIR/hyp_${part}_vad.rttm -c 0.25
    echo "its score detail:"
    echo "DER, MS, FA, SC"
    # pip install git+https://github.com/desh2608/spyder.git@main
    #LC_ALL= spyder  $EXP_DIR/ref_${part}.rttm $EXP_DIR/hyp_${part}_vad.rttm -r single -p -c 0.25
    sctk_dir=SCTK-2.4.12/src/md-eval/
    $sctk_dir/md-eval.pl -c 0.25 -r $EXP_DIR/ref_${part}.rttm -s $EXP_DIR/hyp_${part}_vad.rttm
  done
  #rm $EXP_DIR/ref.rttm $EXP_DIR/ref.rttm
fi
# Evaluating eval VAD output
#its score detail:
#DER, MS, FA, SC
#57.54/13.46/0.55/43.53
#Evaluating test VAD output
#its score detail:
#DER, MS, FA, SC
#52.41/13.44/0.18/38.78
exit 0
