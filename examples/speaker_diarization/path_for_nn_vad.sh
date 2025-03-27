#!/bin/bash

#. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
#. "/mntnfs/lee_data1/maduo/anaconda3/etc/profile.d/conda.sh"
. '/cm/shared/apps/anaconda3/etc/profile.d/conda.sh'
#. "/home/lthpc/anaconda3/etc/profile.d/conda.sh"
conda activate nn_vad

export PYTHONPATH=/mntnfs/lee_data1/maduo/codebase/voice-activity-detection:$PYTHONPATH
