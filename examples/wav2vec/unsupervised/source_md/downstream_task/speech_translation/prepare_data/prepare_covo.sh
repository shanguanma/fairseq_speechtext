#!/usr/local/bin bash

# basic config
#covo_root="root directory of covost (ex. /Drive/cv-corpus-6.1-2020-12-11)"
src_lang=en
tgt_lang=de

stage=-1
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
covo_root=$1   ## e.g.dataset/downstreams_tasks/Speech_Translation/common_voice_corpus_4_en
code_dir=$2  ##  e.g.source_md/downstream_task/speech_translation 
common_voice_4_en_dir=$3
tsv_dir=$covo_root/covost_tsv

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
 echo "download covost code"
 if [ ! -d $covo_root/covost ]; then
    git clone https://github.com/facebookresearch/covost.git $covo_root/covost
 fi
fi
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
  echo "download covost_v2 corpus"
  mkdir $tsv_dir -p
  if [ ! -f $covo_root/covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz ]; then
    wget https://dl.fbaipublicfiles.com/covost/covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz -P $covo_root/
  fi 
tar -zxvf $covo_root/covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz -C $tsv_dir
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
 echo "prepare covost data tsv based on common_voice_4_en"
 python $covo_root/covost/get_covost_splits.py \
    --version 2 --src-lang $src_lang --tgt-lang $tgt_lang \
    --root $tsv_dir \
    --cv-tsv $common_voice_4_en_dir/$src_lang/validated.tsv
fi

# data config
dataset=covost_${src_lang}_${tgt_lang}
data_root=$covo_root
# const
DATA_SRC_KEY=src_text
DATA_TGT_KEY=tgt_text
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 for split in train dev test; do
 #for split in dev test; do
 #for split in test;do
    bash $code_dir/prepare_data/prepare_data.sh --stage 0 --stop-stage  3\
        ${tsv_dir}/covost_v2.${src_lang}_${tgt_lang}.$split.tsv \
        ${covo_root}/${src_lang}/clips/ \
        ${data_root} \
        ${dataset} \
        $split.tsv
 done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 python $code_dir/prepare_data/prepare_gen_fairseq_vocab.py \
    ${data_root}/${dataset}/train.tsv \
    --src-key ${DATA_SRC_KEY} \
    --tgt-key ${DATA_TGT_KEY} \
    --output-dir ${data_root}/${dataset} \
    --model-type char
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 python $code_dir/prepare_data/prepare_create_config.py \
    --sp-model ${data_root}/${dataset}/spm-${DATA_TGT_KEY}.model \
    --vocab-file spm-${DATA_TGT_KEY}.txt \
    --audio-dir $covo_root/${src_lang}/clips \
    --output $data_root/$dataset/config.yaml
fi
