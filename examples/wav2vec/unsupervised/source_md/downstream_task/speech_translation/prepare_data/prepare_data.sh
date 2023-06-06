#!/usr/bin/env bash

# const
DATA_SRC_KEY=src_text
DATA_TGT_KEY=tgt_text
code_dir=source_md/downstream_task/speech_translation

# default argement
stage=0
stop_stage=100
outtsv=test.tsv
. utils/parse_options.sh
srclang=en
tgtlang=de
pathkey=path
srckey=sentence
tgtkey=translation
in_tsv=$1
audio_dir=$2
dataroot=$3
dataset=$4
outtsv=$5
data_dir=${dataroot}/${dataset}
log=${data_dir}/prepare_data.log
out_tsv=${data_dir}/${outtsv}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
 mkdir -p ${data_dir}
 echo "audio_dir: ${audio_dir}"
 echo "srckey : ${srckey}, ${pathkey}"
 echo "[prepare data]" | tee -a $log
 python $code_dir/prepare_data/prepare_data.py ${in_tsv} ${out_tsv}.tmp \
    --verbose \
    -p ${pathkey} \
    -s ${srckey} \
    -t ${tgtkey} \
    -d ${audio_dir} \
    --overwrite \
    | tee -a $log
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
 echo "[clean paired corpus]" | tee -a $log
 python $code_dir/prepare_data/prepare_clean_paired_corpus.py \
    ${out_tsv}.tmp ${out_tsv}.tmp \
    --verbose \
    --min 1 \
    --ratio 5 \
    --overwrite \
    | tee -a $log
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 echo "[clean source text]" | tee -a $log
 python $code_dir/prepare_data/prepare_normalize_tsv.py \
    ${out_tsv}.tmp ${out_tsv}.tmp ${DATA_SRC_KEY}\
    --verbose \
    --normalize \
    --lowercase \
    --remove-punctuation \
    --lang ${srclang} \
    --overwrite \
    | tee -a $log
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 echo "[clean target text]" | tee -a $log
 python $code_dir/prepare_data/prepare_normalize_tsv.py \
    ${out_tsv}.tmp ${out_tsv} ${DATA_TGT_KEY}\
    --verbose \
    --normalize \
    --lang ${tgtlang} \
    --overwrite \
    | tee -a $log

 rm -f ${out_tsv}.tmp
fi
