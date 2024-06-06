#!/usr/bin/env bash
stage=0
stop_stage=100


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  #for part in eval test; do
  for part in test;do
    echo "Running spectral clustering on $part..."
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')

      $train_cmd $EXP_DIR/$part/log/spectral/sc_${filename}.log \
        python clustering_based/spectral/sclust.py \
          --out-rttm-dir $EXP_DIR/$part/spectral \
          --xvec-ark-file $EXP_DIR/$part/xvec/${filename}.ark \
          --segments-file $EXP_DIR/$part/xvec/${filename}.seg \
          --xvec-transform clustering_based/models/ResNet101_16kHz/transform.h5 &
    done
    wait
    )
  done
fi



if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for part in eval test; do
   for name in all overlap nonoverlap;do
    echo "Evaluating $part"
    cat $DATA_DIR/$part/rttm/*.rttm > $EXP_DIR/ref_${part}_spectral.rttm
    cat $EXP_DIR/$part/spectral/*.rttm > $EXP_DIR/hyp_${part}_spectral.rttm
    #$sctk_dir/md-eval.pl -r exp/ref.rttm -s exp/hyp.rttm -c 0.25|\
    #  awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
    echo "its score detail:"
    # pip install git+https://github.com/desh2608/spyder.git@main
    echo "in $name mode"
    LC_ALL= spyder  $EXP_DIR/ref_${part}_spectral.rttm $EXP_DIR/hyp_${part}_spectral.rttm -r $name -c 0.25
    #sctk_dir=SCTK-2.4.12/src/md-eval/
    #$sctk_dir/md-eval.pl -c 0.25 -r $EXP_DIR/ref_${part}_spectral.rttm -s $EXP_DIR/hyp_${part}_spectral.rttm 
 
  done
 done
fi
# result: 
# DER, MS, FA, SC 
# Evaluating eval
#its score detail:
#15.24/13.46/0.55/1.23
#Evaluating test
#its score detail:
#16.15/13.44/0.18/2.53

## Evaluating eval
#its score detail:
#in all mode
#Evaluated 9 recordings on `all` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ Overall     │       11758.99 │  13.29% │      0.68% │   1.19% │ 15.16% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
#Evaluating eval
#its score detail:
#in overlap mode
#Evaluated 9 recordings on `overlap` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ Overall     │        2782.96 │  53.76% │      0.00% │   1.06% │ 54.83% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
#Evaluating eval
#its score detail:
#in nonoverlap mode
#Evaluated 9 recordings on `nonoverlap` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │   DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
#│ Overall     │        8976.03 │   0.74% │      0.89% │   1.23% │ 2.86% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛
#Evaluating test
#its score detail:
#in all mode
#Evaluated 21 recordings on `all` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ Overall     │       30309.37 │  13.17% │      0.22% │   2.53% │ 15.93% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
#Evaluating test
#its score detail:
#in overlap mode
#Evaluated 21 recordings on `overlap` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ Overall     │        6848.25 │  55.03% │      0.00% │   1.55% │ 56.58% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
#Evaluating test
#its score detail:
#in nonoverlap mode
#Evaluated 21 recordings on `nonoverlap` regions.
#DER metrics:
#╒═════════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │   DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
#│ Overall     │       23461.12 │   0.96% │      0.28% │   2.82% │ 4.06% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛
exit 0
