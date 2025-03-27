#!/usr/bin/env bash
stage=0
stop_stage=100
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DATA_DIR=data/manifests/alimeeting
EXP_DIR=exp/alimeeting_cluster_base
mkdir -p exp

model_dir=

# Duo MA check and make sure the below information.
#This xvector model, plda and transform.h5 is from https://github.com/BUTSpeechFIT/VBx/tree/master/VBx/models/ResNet101_16kHz
#Both xvector model and plda are trained on the voxceleb1,voceleb2 and cnceleb. this information is from the paper("Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks")

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for part in eval test; do
    echo "Extracting x-vectors for ${part}..."
    mkdir -p $EXP_DIR/$part/xvec
    (
    for audio in $(ls $DATA_DIR/${part}/audios/*.wav | xargs -n 1 basename)
    do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list_${filename}.txt

      $cuda_cmd $EXP_DIR/${part}/log/xvec/xvec_${filename}.log \
        python clustering_based/xvector/predict.py \
          --gpus true \
          --in-file-list exp/list_${filename}.txt \
          --in-lab-dir $EXP_DIR/${part}/vad \
          --in-wav-dir $DATA_DIR/${part}/audios \
          --out-ark-fn $EXP_DIR/${part}/xvec/${filename}.ark \
          --out-seg-fn $EXP_DIR/${part}/xvec/${filename}.seg \
          --model ResNet101 \
          --weights  clustering_based/models/ResNet101_16kHz/nnet/final.onnx \
          --backend onnx &
    done
    wait
    )
    rm exp/list_*
  done
fi

exit 0
