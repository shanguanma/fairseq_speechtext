#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

### noly for sribd class
## build c++ part using CUDA, however in head node of this slurm server system hasn't cuda gpu.
#cd /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
#pip install ninja ## using fast  distutils backend training for pytorch, it is very important
#pip install --editable ./  ## for python package, it can be installed at local environment
#python setup.py build_ext --inplace
## check ninja  it works correctly ?
##  if return exit code 0, it should works correctly, otherwise  it doesn't work
ninja --version  | echo $?
##  you can reinstall via pip uninstall -y ninja && pip install ninja


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm\
            hydra.job.name=$exp_dir/pretrain
### 4V100: training about  day
###           200steps: about  minites
fi
