#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    echo "prepared 1h ,9h finetune data format (train-1h.tsv, train-9h.tsv)"
    data_wav_dir=/workspace2/maduo/dataset/librispeech_finetune/librispeech_finetuning
    tsv_dir=/workspace2/maduo/dataset/format/librispeech/
    for name in 1h 9h;do
       python source_md/wav2vec-u2/wav2vec_manifest_md.py \
            $data_wav_dir/$name\
            --dest_file $tsv_dir/train-${name}.tsv\
            --ext flac
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    echo "prepared 1h, 9h finetune data format (train-1h.ltr, train-9h.ltr)"
    data_wav_dir=/workspace2/maduo/dataset/librispeech_finetune/librispeech_finetuning
    tsv_dir=/workspace2/maduo/dataset/format/librispeech/
    for name in 1h 9h;do
       python source_md/wav2vec-u2/text/libri_labels.py  \
            $tsv_dir/train-$name.tsv\
            --output-dir $tsv_dir\
            --output-name train-$name

    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "prepared 10h finetune data format (train-10h.tsv,)"
    data_wav_dir=/workspace2/maduo/dataset/librispeech_finetune/librispeech_finetuning
    tsv_dir=/workspace2/maduo/dataset/format/librispeech/

    python source_md/wav2vec-u2/two_tsv2one.py   \
            $tsv_dir/train-1h.tsv\
            $tsv_dir/train-9h.tsv\
            $tsv_dir/train-10h.tsv
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    echo "prepared 10h finetune data format (train-10h.ltr,)"
    data_wav_dir=/workspace2/maduo/dataset/librispeech_finetune/librispeech_finetuning
    tsv_dir=/workspace2/maduo/dataset/format/librispeech/
    cat $tsv_dir/train-1h.ltr $tsv_dir/train-9h.ltr > $tsv_dir/train-10h.ltr
    cat $tsv_dir/train-1h.wrd $tsv_dir/train-9h.wrd > $tsv_dir/train-10h.wrd
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
    echo "prepared 10mins finetune data format (train-10mins.tsv),"
    echo "I select first 10mins from 1h folder"
    data_wav_dir=/workspace2/maduo/dataset/librispeech_finetune/librispeech_finetuning
    tsv_dir=/workspace2/maduo/dataset/format/librispeech/
    for name in 1h;do
       python source_md/wav2vec-u2/wav2vec_manifest_md.py \
            $data_wav_dir/$name/0\
            --dest_file $tsv_dir/train-10mins.tsv\
            --ext flac
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    echo "prepared 10mins finetune data format (train-10mins.ltr),"
    echo "I select first 10mins from 1h folder"
    data_wav_dir=/workspace2/maduo/dataset/librispeech_finetune/librispeech_finetuning
    tsv_dir=/workspace2/maduo/dataset/format/librispeech/
       python source_md/wav2vec-u2/text/libri_labels.py  \
            $tsv_dir/train-10mins.tsv\
            --output-dir $tsv_dir\
            --output-name train-10mins


fi




