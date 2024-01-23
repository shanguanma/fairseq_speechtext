#!/usr/bin/env bash
  
stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


####
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared unpaired speech and unpaired text label folder"
   ## in order to get number of text utterances equal to number of wav utterances in sthubert method
   ### speech part
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/train-960.km
   # 281241 dataset/format/librispeech/mfcc/mfcc_lab/train-960.km
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/dev-other.km
   # 2864 dataset/format/librispeech/mfcc/mfcc_lab/dev-other.km
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/dev-clean.km
   # 2703 dataset/format/librispeech/mfcc/mfcc_lab/dev-clean.km

   #### text part
   ### wc -l  dataset/format/librispeech/librispeech_lm_monophncode_using_monophn_dict/librilm.phncode
   ### 33390030 dataset/format/librispeech/librispeech_lm_monophncode_using_monophn_dict/librilm.phncode
   root_dir=dataset/format/librispeech
   textdata_dir=$root_dir/librispeech_lm_monophncode_using_monophn_dict
   dest_dir=$root_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   mkdir -p $dest_dir
   sed -n "30000,311240p" $textdata_dir/librilm.phncode > $dest_dir/train-960.textphncode
   sed -n "320000,322863p" $textdata_dir/librilm.phncode > $dest_dir/dev-other.textphncode
   sed -n "330000,332702p" $textdata_dir/librilm.phncode > $dest_dir/dev-clean.textphncode
   #sed -n "330000,335405p"   $textdata_dir/librilm.phncode > $dest_dir/dev-clean-wav.textphncode  
   speechdata_dir=$root_dir/librispeech_frame_monophncode_using_wav2vec-u2_model
   cp -r $speechdata_dir/{train-960,dev-other,dev-clean}.phncode $dest_dir
   mv $dest_dir/train-960.phncode $dest_dir/train-960.speechphncode
   mv $dest_dir/dev-other.phncode $dest_dir/dev-other.speechphncode
   mv $dest_dir/dev-clean.phncode $dest_dir/dev-clean.speechphncode
   cp -r dataset/format/librispeech/librispeech_lm_norm_phn_seq/phoness/dict.phncode.txt  $dest_dir/
   mv $dest_dir/dict.phncode.txt $dest_dir/dict.speechphncode.txt
   cp -r dataset/format/librispeech/librispeech_lm_norm_phn_seq/phoness/dict.phncode.txt  $dest_dir/
   mv $dest_dir/dict.phncode.txt $dest_dir/dict.textphncode.txt
   echo "display all ....."
   wc -l $dest_dir/*

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "pretrain hubert on wav2vec-u2.0 15 layer pesudo label, label_rate=50"
   echo "training on 400k steps for train-960 of librispeech unpair speech and librispeech_lm unpaired text"
   echo "the above unpaired speech and unpaired text are coverted into 41 phncode including silence phncode"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model # ##postfix *.speechphncode *.textphncode files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode_and_librilm_monophncode

   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_base_librispeech2\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain

## when it is running 176679 steps, correct_m_0 and  correct_u_0  are very fast reducing
# 在178ksteps : train correct_m 从0.7595掉到0.2646 并一直到200k,因此我终止实验了

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "fine tune sthubert model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=source_md/wav2vec-u2/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name sthubert_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/finetune

## when it is running to epoch 97
## File "/workspace2/maduo/fairseq_speechtext/fairseq/optim/dynamic_loss_scaler.py", line 61, in check_overflow
##    raise FloatingPointError(
##FloatingPointError: Minimum loss scale reached (0.0001). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name sthubert_infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:400, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   #  

fi

# reduce steps number
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   
   echo "pretrain hubert on wav2vec-u2.0 15 layer pesudo label, label_rate=50"
   echo "training on 400k steps for train-960 of librispeech unpair speech and librispeech_lm unpaired text"
   echo "the above unpaired speech and unpaired text are coverted into 41 phncode including silence phncode"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model # ##postfix *.speechphncode *.textphncode files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_base_librispeech_monophone\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
fi
#
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "fine tune sthubert model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=source_md/wav2vec-u2/
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_372_150000.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/finetune
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name sthubert_infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:150k@372epochs, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   # 7.23           16.43        7.37        16.53

fi





if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then

   echo "pretrain hubert on wav2vec-u2.0 15 layer pesudo label, label_rate=50"
   echo "training on 400k steps for train-960 of librispeech unpair speech and librispeech_lm unpaired text"
   echo "the above unpaired speech and unpaired text are coverted into 41 phncode including silence phncode"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model # ##postfix *.speechphncode *.textphncode files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_base_librispeech\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
fi
## check pretrain model name
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "fine tune sthubert model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=source_md/wav2vec-u2/
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h_for_hubert \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_289_400000.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:400k@372epochs, finetune:80k@222epchs
   # ## without lm
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   #  4.8454       11.4734     4.9613        11.2052 
   # with 4-gram lm
   
   

fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech with 4-gram lm "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h_with_4gram
   testsets="dev-clean dev-other test-clean test-other"
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/workspace2/maduo/dataset/librispeech/kenlm_files/4-gram.arpa  ## word lm
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h_with_4gram\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                common_eval.quiet=true

   #result: mfcc iter:400k@372epochs, finetune:80k@222epchs
   # ## without lm
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   #  4.8454       11.4734     4.9613        11.2052 
   # with 4-gram lm
   #  2.55          7.16        3.10          7.37        
   done
fi


#### 2023.5.17
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "prepared unpaired speech and unpaired text label folder via add four specify symbols, it should be correct way"
   ## in order to get number of text utterances equal to number of wav utterances in sthubert method
   ### speech part
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/train-960.km
   # 281241 dataset/format/librispeech/mfcc/mfcc_lab/train-960.km
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/dev-other.km
   # 2864 dataset/format/librispeech/mfcc/mfcc_lab/dev-other.km
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/dev-clean.km
   # 2703 dataset/format/librispeech/mfcc/mfcc_lab/dev-clean.km

   #### text part
   ### wc -l  dataset/format/librispeech/librispeech_lm_monophncode_using_monophn_dict/librilm.phncode
   ### 33390030 dataset/format/librispeech/librispeech_lm_monophncode_using_monophn_dict/librilm.phncode
   root_dir=dataset/format/librispeech
   textdata_dir=$root_dir/librispeech_lm_monophncode_using_monophn_dict_newer
   dest_dir=$root_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model_newer
   mkdir -p $dest_dir
   sed -n "30000,311240p" $textdata_dir/librilm.phncode > $dest_dir/train-960.textphncode
   sed -n "320000,322863p" $textdata_dir/librilm.phncode > $dest_dir/dev-other.textphncode
   sed -n "330000,332702p" $textdata_dir/librilm.phncode > $dest_dir/dev-clean.textphncode
   sed -n "330000,335405p"   $textdata_dir/librilm.phncode > $dest_dir/dev-clean-wav.textphncode # for huawei server
   speechdata_dir=$root_dir/librispeech_frame_monophncode_using_wav2vec-u2_model_newer
   cp -r $speechdata_dir/{train-960,dev-other,dev-clean}.unsupphncode $dest_dir
   mv $dest_dir/train-960.unsupphncode $dest_dir/train-960.speechphncode
   mv $dest_dir/dev-other.unsupphncode $dest_dir/dev-other.speechphncode
   mv $dest_dir/dev-clean.unsupphncode $dest_dir/dev-clean.speechphncode
   cp -r $speechdata_dir/dict.unsupphncode.txt  $dest_dir/
   mv $dest_dir/dict.unsupphncode.txt $dest_dir/dict.speechphncode.txt
   cp -r  $speechdata_dir/dict.unsupphncode.txt  $dest_dir/
   mv $dest_dir/dict.unsupphncode.txt $dest_dir/dict.textphncode.txt
   echo "display all ....."
   wc -l $dest_dir/*


   ### 2023.8.31 add train-clean-100.textphncode and dev-clean.ltr,dev-other.ltr,train-clean-100.ltr for voicelm2 finetune
   cat  dataset/format/librispeech/librispeech_lm_monophncode_using_monophn_dict/librilm.phncode  | shuf |head -n 28539 > $dest_dir/train-clean-100.textphncode
   cp -r dataset/format/librispeech/{dev-clean.ltr,dev-other.ltr,train-clean-100.ltr} dataset/format/librispeech/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech with 4-gram lm "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h_with_4gram
   testsets="dev-clean dev-other test-clean test-other"
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst
   path_to_lm=/workspace2/maduo/dataset/librispeech/kenlm_files/4-gram.arpa
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=1  python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name sthubert_infer_kenlm\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h_with_4gram\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                common_eval.quiet=true 
             
                 

   done
   #result: mfcc iter:150k@372epochs, finetune:80k@222epchs
   ## without lm
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   # 7.23           16.43        7.37        16.53
   ## with 4-gram lm 
   ## 3.2241        9.3200       3.7457      9.6615   

fi
## fairseqlm maybe not apply base pretrain model,its wer is 100% because public pretrain base model only is apply at 4-gram lm  so I give up it
if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech with transformer lm(fairseqlm from wav2letter) "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h_with_fairseqlm
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="dev-clean"
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst
   path_to_lm=/workspace2/maduo/dataset/librispeech/fairseqlm_files/lm_librispeech_word_transformer.pt
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=1  python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name sthubert_infer_fsqlm\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h_with_fairseqlm\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=fairseqlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.lmweight=0.1 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                decoding.beam=1500 \
                common_eval.quiet=false



   done
   #result: mfcc iter:150k@372epochs, finetune:80k@222epchs
   ## without lm
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   # 7.23           16.43        7.37        16.53
   ## with 4-gram lm
   ## 3.2241        9.3200       3.7457      9.6615

fi


## add ctc loss for text part
## add swap embedding for speech part, and compute mask lm loss twice.
## it is running at huawei server.
## for debug
if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ];then

   echo "pretrain hubert on wav2vec-u2.0 15 layer pesudo label, label_rate=50"
   echo "training on 150k steps for train-960 of librispeech unpair speech and librispeech_lm unpaired text"
   echo "the above unpaired speech and unpaired text are coverted into 41 phncode including silence phncode"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model # ##postfix *.speechphncode *.textphncode files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   #model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode_speechlm_style
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode_speechlm_style1
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=2
   update_freq=16
   CUDA_VISIBLE_DEVICES=0,4 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_base_librispeech_monophone_speechlm_style\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
fi


## check pretrain model name
if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
   echo "fine tune sthubert model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=source_md/wav2vec-u2/
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode_speechlm_style1
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h_for_sthubert2 \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi

## add d2v style loss for debug 
if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ];then

   echo "pretrain hubert on wav2vec-u2.0 15 layer pesudo label, label_rate=50"
   echo "training on 150k steps for train-960 of librispeech unpair speech and librispeech_lm unpaired text"
   echo "the above unpaired speech and unpaired text are coverted into 41 phncode including silence phncode"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model # ##postfix *.speechphncode *.textphncode files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   #model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode_speechlm_style
   model_name=pretrain_on_base_sthubert_4gpu_8update_150k_w2vu2_librispeech_monophncode_and_librilm_monophncode_speechlm_style_and_d2v_style
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=2
   update_freq=16
   CUDA_VISIBLE_DEVICES=1,2 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_base_librispeech_monophone_speechlm_style_and_d2v_style\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
fi



