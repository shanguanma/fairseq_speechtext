#!/usr/bin/env bash


stage=0
stop_stage=1000

. utils/parse_options.sh
#. path_for_nn_vad.sh
. path_for_fsq_sptt.sh



if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
  echo "spectral clustering ...."
  python scripts/magicdata-RAMC/020_spectral_cluster.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "cpu get embedding"
	python scripts/magicdata-RAMC/020_spectral_cluster_v2.py
   echo "finish"
fi
