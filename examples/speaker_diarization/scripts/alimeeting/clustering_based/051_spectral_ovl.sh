#!/usr/bin/env bash
stage=0
stop_stage=100


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  for part in eval test; do
    echo "Running spectral clustering on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')

      $train_cmd $EXP_DIR/$part/log/spectral_ovl/sc_${filename}.log \
        python clustering_based/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/$part/spectral_ovl \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
	  --overlap-rttm $EXP_DIR/$part/ovl/${filename}.rttm \
          --xvec-transform clustering_based/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi



if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for part in eval test; do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > $EXP_DIR/ref_${part}_spectral_ovl.rttm
    cat $EXP_DIR/$part/spectral_ovl/*.rttm > $EXP_DIR/hyp_${part}_spectral_ovl.rttm
    #$sctk_dir/md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25|\
    #  awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
    echo "its score detail:"
    echo "DER, MS, FA, SC"
    # pip install git+https://github.com/desh2608/spyder.git@main
    #LC_ALL= spyder  $EXP_DIR/ref_${part}_spectral_ovl.rttm $EXP_DIR/hyp_${part}_spectral_ovl.rttm -r single -p -c 0.25
    sctk_dir=SCTK-2.4.12/src/md-eval/
    $sctk_dir/md-eval.pl -c 0.25 -r $EXP_DIR/ref_${part}_spectral_ovl.rttm -s $EXP_DIR/hyp_${part}_spectral_ovl.rttm 
 done
fi
# Evaluating eval
#its score detail:
#DER, MS, FA, SC
#15.19/13.46/0.55/1.18
#Evaluating test
#its score detail:
#DER, MS, FA, SC
#16.67/13.44/0.18/3.04
exit 0
