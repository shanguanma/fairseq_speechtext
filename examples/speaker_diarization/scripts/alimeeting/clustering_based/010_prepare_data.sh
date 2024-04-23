#!/usr/bin/env bash
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

CORPUS_DIR=/data/alimeeting/
DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base

mkdir -p exp

if [ $stage -le 0 ]; then
  echo "Preparing AliMeeting data..."
  $train_cmd $EXP_DIR/log/prepare.log \
    python clustering_based/prepared_alimeeting_lhotse_format.py --data-dir $CORPUS_DIR --output-dir $DATA_DIR
fi

exit 0
