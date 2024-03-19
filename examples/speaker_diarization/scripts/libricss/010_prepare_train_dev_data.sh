#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

# LibriCSS is a multi-talker meeting corpus formed from mixing together LibriSpeech utterances
# and replaying in a real meeting room. It consists of 10 1-hour sessions of audio, each
# recorded on a 7-channel microphone. The sessions are recorded at a sampling rate of 16 kHz.
# For more information, refer to the paper:
# Z. Chen et al., "Continuous speech separation: dataset and analysis,"
# ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
# Barcelona, Spain, 2020


# When libricss is as speaker diarization task
# dev set and eval set is determined base on section V-A of `TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings` and 
# section 2 of `Integration of speech separation, diarization, and recognition for multi-speaker meetings: System description, comparison, and analysis`

# train set, we will follow this script (https://github.com/shanguanma/jsalt2020_simulate/blob/master/docs/mtgsim.md.) 
#                          base on the paper: section V-A of `TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings`
#                                             `Integration of speech separation, diarization, and recognition for multi-speaker meetings: System description, comparison, and analysis`



## libricss is as speaker diarization task , its reference paper: https://arxiv.org/pdf/2309.16482.pdf
##                                                                https://arxiv.org/pdf/2110.03151.pdf
##                                                                https://arxiv.org/pdf/2303.03849.pdf
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then

   data_path=/data/alimeeting
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   echo " Process dataset: Eval dataset, get json files"
   python $fairseq_dir/examples/speaker_diarization/scripts/prepare_data.py \
    --data_path ${data_path} \
    --type Eval \

fi
