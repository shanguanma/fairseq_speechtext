#!/bin/bash
  

stage=0

stop_stage=1000
nj=32
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
RVAD_ROOT=/mntnfs/lee_data1/maduo/codebase/rVADfast
fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "get vad file"
   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   datasets="train_m"
   mkdir -p $des_dir
   mkdir -p $des_dir/parallel
   for name in $datasets;do
     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/vads_for_wavscp.py\
             -r $RVAD_ROOT < $input_dir/m/$name/wav.scp > $des_dir/${name}.vads



    done
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "get remove silence audio file with multi cpus"
   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   datasets="train_m"
   #mkdir -p $des_dir/parallel
   for name in $datasets;do
   #export  OMP_NUM_THREADS=1
    ## debug:
    #  torchrun --nproc_per_node=5 --master_port=12345 codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/vads_for_wavscp_parallel.py -r codebase/rVADfast --output_dir tests --wav_scp tests/wav5.scp 
  # it is very slow, Even when using parallel
  #torchrun --nproc_per_node=$nj --master_port=12345 \
  #     $fairseq_dir/examples/wav2vec/unsupervised/scripts/vads_for_wavscp_parallel.py \
  #     -r $RVAD_ROOT --output_dir $des_dir/parallel --wav_scp $input_dir/m/$name/wav.scp

  torchrun --nproc_per_node=$nj --master_port=12345 \
      $fairseq_dir/examples/wav2vec/unsupervised/scripts/silero-vad.py \
      --wavscp $input_dir/m/$name/wav.scp  --out $des_dir/base_on_silero-vad_onnx_torchrun_parallel

   done
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepare tsv file for no silence wenetspeech"
   raw_wav_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   #datasets="dev-clean"
   #datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   datasets="base_on_silero-vad_onnx_torchrun_parallel"
   for name in $datasets;do
     python3  source-md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name.tsv\
           --ext opus
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done

fi
## running at hltsz cluster
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "get Chinese hubert large model representation of no silence wenetspeech"
   export PYTHONPATH=codebase/fairseq_speechtext:$PYTHONPATH
   #echo "get wav2vec_large feature for librispeech"
   #des_dir=/workspace2/maduo/dataset/format/librispeech
   des_dir=/home/maduo/dataset/wenetspeech_no_silence/
   feat_dir=/home/maduo/dataset/wenetspeech_no_silence/hubet_large_feat_dir_no_silence
   #model=/workspace2/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   model=/home/maduo/model_hub/chinese-hubert-large-fairseq-ckpt.pt 
   mkdir -p $feat_dir
   layer=14 #0-based index
   for name in base_on_silero-vad_onnx_torchrun_parallel ;do
     python  source_md/wav2vec-u2/hubert_extract_features.py\
           $des_dir\
           --split $name\
           --save-dir $feat_dir\
           --checkpoint $model\
           --layer $layer\
	   --wavfile-suffix ".tsv"
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done

fi

