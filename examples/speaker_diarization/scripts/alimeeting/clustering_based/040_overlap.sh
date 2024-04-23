#!/usr/bin/env bash
stage=0
stop_stage=100

# Overlap detector Hyperparameters (tuned on dev)
onset=0.5
offset=0.6
min_duration_on=0.4
min_duration_off=0.5

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh




DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base

mkdir -p exp

model_dir=/home/maduo/model_hub/vad/pyannote_segmentation/
sctk_dir=SCTK-2.4.12/src/md-eval/

## step 1: download pyannote segmentation model
## step 2: resegmentation on target dataset (i.e. alimeeting) , called as segmentation finetune
## step 3: get target dataset overlap vad using the above segmentation model.
if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then

  if [ -f  $model_dir/alimeeting_epoch0_step2600.ckpt ]; then
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
    python clustering_based/pyannote/train_seg_finetune.py \
            --dataset AliMeeting \
            --exp_dir exp/pyannote/alimeeting\
            --pyannote_segmentation_model $model_dir/pytorch_model.bin
    #cp exp/pyannote/alimeeting/lightning_logs/version_0/checkpoints/epoch=0-step=2492.ckpt $model_dir/alimeeting_epoch0_step2492.ckpt
  fi
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   cp exp/pyannote/alimeeting/lightning_logs/version_3409/checkpoints/epoch\=0-step\=2600.ckpt $model_dir/alimeeting_epoch0_step2600.ckpt
  for part in eval test; do
    echo "Running overlap detection on $part set..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt
      
      $train_cmd $EXP_DIR/${part}/log/ovl/ovl_${filename}.log \
        python clustering_based/overlap/pyannote_overlap.py \
          --model $model_dir/alimeeting_epoch0_step2600.ckpt \
          --in-dir $DATA_DIR/${part}/audios \
          --file-list exp/list_${filename}.txt \
          --out-dir $EXP_DIR/${part}/ovl \
          --onset ${onset} --offset ${offset} \
          --min-duration-on ${min_duration_on} \
          --min-duration-off ${min_duration_off} & 
    done
    wait
    )
    rm exp/list_*.txt
  done
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Onset: $onset Offset: $offset Min_duration_on: $min_duration_on Min_duration_off: $min_duration_off"
  for part in eval test; do
    echo "Evaluating ${part} overlap detector output"
    cat $DATA_DIR/${part}/rttm/* | clustering_based/get_overlap_segments.py | grep overlap > $EXP_DIR/ref_${part}_overlap.rttm
    cat $EXP_DIR/${part}/ovl/*.rttm > $EXP_DIR/hyp_${part}_overlap.rttm 
    #$sctk_dir/md-eval.pl -r $EXP_DIR/ref_${part}_overlap.rttm -s $EXP_DIR/hyp_${part}_overlap.rttm -c 0.25 |\
    #  awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
    echo "its score detail:"
    echo "DER, MS, FA, SC"
    # pip install git+https://github.com/desh2608/spyder.git@main
    #LC_ALL= spyder  $EXP_DIR/ref_${part}_overlap.rttm $EXP_DIR/hyp_${part}_overlap.rttm -r single -p -c 0.25
    sctk_dir=SCTK-2.4.12/src/md-eval/
    $sctk_dir/md-eval.pl -c 0.25 -r $EXP_DIR/ref_${part}_overlap.rttm -s $EXP_DIR/hyp_${part}_overlap.rttm 
done
fi
# Onset: 0.5 Offset: 0.6 Min_duration_on: 0.4 Min_duration_off: 0.5
#Evaluating eval overlap detector output
#its score detail:
#DER, MS, FA, SC
#32.41/27.77/4.63/0.00
#Evaluating test overlap detector output
#its score detail:
#DER, MS, FA, SC
#35.59/27.19/8.40/0.00
#
exit 0
# Onset: 0.5 Offset: 0.6 Min_duration_on: 0.4 Min_duration_off: 0.5
#Evaluating eval overlap detector output
#its score detail:
#Evaluated 9 recordings on `single` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ R8001_M8004 │         309.62 │  28.92% │      0.00% │   0.00% │ 28.92% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8003_M8001 │         184.38 │  37.68% │      0.00% │   0.00% │ 37.68% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8007_M8010 │         796.02 │  21.01% │      0.00% │   0.00% │ 21.01% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8007_M8011 │         273.51 │  20.25% │      0.00% │   0.00% │ 20.25% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8008_M8013 │         154.70 │  52.02% │      0.00% │   0.00% │ 52.02% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8018 │         174.90 │  89.06% │      0.00% │   0.00% │ 89.06% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8019 │         690.79 │  89.68% │      0.00% │   0.00% │ 89.68% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8020 │         418.81 │  94.20% │      0.00% │   0.00% │ 94.20% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ Overall     │        3002.73 │  54.35% │      0.00% │   0.00% │ 54.35% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
#Evaluating test overlap detector output
#its score detail:
#Evaluated 21 recordings on `single` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ R8002_M8002 │         276.33 │  31.60% │      0.00% │   0.00% │ 31.60% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8002_M8003 │         347.66 │  52.89% │      0.00% │   0.00% │ 52.89% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8004_M8005 │         360.89 │  21.29% │      0.00% │   0.00% │ 21.29% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8004_M8006 │         899.93 │  19.34% │      0.00% │   0.00% │ 19.34% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8005_M8007 │         628.13 │  23.19% │      0.00% │   0.00% │ 23.19% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8005_M8008 │         290.27 │  16.28% │      0.00% │   0.00% │ 16.28% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8005_M8009 │         924.73 │  25.17% │      0.00% │   0.00% │ 25.17% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8006_M8012 │         333.17 │  20.47% │      0.00% │   0.00% │ 20.47% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8008_M8014 │         101.39 │  46.61% │      0.00% │   0.00% │ 46.61% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8008_M8015 │         126.13 │  63.81% │      0.00% │   0.00% │ 63.81% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8008_M8016 │         117.34 │  68.01% │      0.00% │   0.00% │ 68.01% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8008_M8017 │         151.31 │  91.22% │      0.00% │   0.00% │ 91.22% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8021 │         177.62 │  63.66% │      0.00% │   0.00% │ 63.66% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8022 │         269.18 │  88.58% │      0.00% │   0.00% │ 88.58% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8023 │         512.90 │  94.38% │      0.00% │   0.00% │ 94.38% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8024 │         441.06 │  97.97% │      0.00% │   0.00% │ 97.97% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8025 │         671.09 │  95.59% │      0.00% │   0.00% │ 95.59% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8026 │         349.23 │  95.62% │      0.00% │   0.00% │ 95.62% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8027 │         541.24 │  98.10% │      0.00% │   0.00% │ 98.10% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ R8009_M8028 │         429.41 │  94.39% │      0.00% │   0.00% │ 94.39% │
#├─────────────┼────────────────┼─────────┼────────────┼─────────┼────────┤
#│ Overall     │        7949.01 │  57.13% │      0.00% │   0.00% │ 57.13% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
