#!/usr/bin/env bash
stage=0
stop_stage=100

# Hyperparameters (same as AISHELL-4)
Fa=0.5
Fb=40
loopP=0.9

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base

mkdir -p exp

model_dir=/home/maduo/model_hub/vad/pyannote_segmentation/
sctk_dir=SCTK-2.4.12/src/md-eval/

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for part in eval test; do
    echo "Running VBx with Fa=$Fa, Fb=$Fb, loopP=$loopP on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      
      $train_cmd $EXP_DIR/$part/log/vbx/vb_${filename}.log \
        python clustering_based/vbx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir $EXP_DIR/$part/vbx \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --xvec-transform clustering_based/models/ResNet101_16kHz/transform.h5 \
          --plda-file clustering_based/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --init-smoothing 7.0 \
          --lda-dim 128 \
          --Fa $Fa \
          --Fb $Fb \
          --loopP $loopP &
    done
    wait
    )
  done
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for part in eval test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > $EXP_DIR/ref_${part}_vbx.rttm
    cat $EXP_DIR/$part/vbx/*.rttm > $EXP_DIR/hyp_${part}_vbx.rttm
    #$sctk_dir/md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25|\
    #  awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
    echo "its score detail:"
    echo "DER, MS, FA, SC"
    # pip install git+https://github.com/desh2608/spyder.git@main
    #LC_ALL= spyder  $EXP_DIR/ref_${part}_vbx.rttm $EXP_DIR/hyp_${part}_vbx.rttm -r single -p -c 0.25
    sctk_dir=SCTK-2.4.12/src/md-eval/
    $sctk_dir/md-eval.pl -c 0.25 -r $EXP_DIR/ref_${part}_vbx.rttm -s $EXP_DIR/hyp_${part}_vbx.rttm
done
fi
# Evaluating eval
#its score detail:
#DER, MS, FA, SC
#16.34/13.46/0.55/2.32
#Evaluating test
#its score detail:
#DER, MS, FA, SC
#16.67/13.44/0.18/3.04 
exit 0
