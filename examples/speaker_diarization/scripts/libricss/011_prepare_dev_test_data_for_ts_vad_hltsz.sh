#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
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

# dev set and test set, we will follow those papers setting
#                                              section V-A of `TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings`
#                                             `Integration of speech separation, diarization, and recognition for multi-speaker meetings: System description, comparison, and analysis`



## libricss is as speaker diarization task , its reference paper: https://arxiv.org/pdf/2309.16482.pdf
##                                                                https://arxiv.org/pdf/2110.03151.pdf
##                                                                https://arxiv.org/pdf/2303.03849.pdf



