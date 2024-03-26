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

# train set, we will follow this script (https://github.com/shanguanma/jsalt2020_simulate/blob/master/docs/mtgsim.md.) 
#                          base on the paper: section V-A of `TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings`
#                                             `Integration of speech separation, diarization, and recognition for multi-speaker meetings: System description, comparison, and analysis`



## libricss is as speaker diarization task , its reference paper: https://arxiv.org/pdf/2309.16482.pdf
##                                                                https://arxiv.org/pdf/2110.03151.pdf
##                                                                https://arxiv.org/pdf/2303.03849.pdf
## reference: https://github.com/shanguanma/jsalt2020_simulate/blob/master/scripts/preprocess.sh
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "Convert FLAC files to WAV via soundfile packkage for LibriSpeech"
   srcdir=/data/LibriSpeech
   dstdir=/data/SimLibriCSS
   mkdir -p $dstdir

   for name  in train dev test; do 
    mkdir -p $dstdir/$name
    if [ "$name" == train ]; then
        python source_md/convert_flac_to_wav.py --srcdir $srcdir/train-clean-100 $srcdir/train-clean-360 $srcdir/train-other-500 --dstdir $dstdir/wav
    else
        python source_md/convert_flac_to_wav.py --srcdir $srcdir/${name}-clean --dstdir $dstdir/wav
    fi
   done
   
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate wavlist"
   
   srcdir=/data/LibriSpeech
   #dstdir=/data/SimLibriCSS
   gen_cmd=source_md/run.pl 
   #splitdir=$dstdir/filelist/split${nj}
   #mkdir -p ${splitdir}/log

   #for name  in train dev test; do
   for name  in train ; do
    # List the original wav files.
    dstdir=/data/SimLibriCSS/$name
    mkdir -p $dstdir
    mkdir -p $dstdir/filelist
    if [ "$name" == train ]; then
        python  source_md/gen_filelist.py --srcdir $dstdir/wav/train-clean-100 $dstdir/wav/train-clean-360 $dstdir/wav/train-other-500 --outlist $dstdir/filelist/${name}.list
    else
        python source_md/gen_filelist.py  --srcdir $dstdir/wav/${name}-clean --outlist $dstdir/filelist/${name}.list
    fi        

    # Split trainlist for parallel processing
    #source_md/split_scp.pl ${dstdir}/filelist/${name}.list $(for j in $(seq ${nj}); do echo ${splitdir}/${name}.${j}.list; done)

    # Remove silence regions. This allows us to accurately control the overlap ratio distribution duing training.
    #${gen_cmd} JOB=1:${nj} ${splitdir}/log/tight_segment.JOB.log \
    #    python source_md/tight_segment.py --inputlist ${splitdir}/${name}.JOB.list --outputdir ${dstdir}/wav_newseg
    done
fi


# reference: https://github.com/shanguanma/jsalt2020_simulate/blob/master/scripts/run_meetings.sh
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  for name  in train dev test; do
   tgtroot=/data/SimLibriCSS/SimLibriCSS-$name
   srcdir=/data/SimLibriCSS/$name/wav
   # List the source files. 
   datalist=$tgtroot/$name.list
   python source_md/gen_filelist.py --srcdir $srcdir --outlist $datalist

   # Split datalist for parallel processing
   splitdir=${tgtroot}/split${nj}
   mkdir -p ${splitdir}/log

   source_md/split_scp.pl ${datalist} $(for j in $(seq ${nj}); do echo ${splitdir}/${name}.${j}.list; done)
 done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  gen_cmd=source_md/run.pl
  for name  in train dev test; do
   tgtroot=/data/SimLibriCSS/SimLibriCSS-$name
   srcdir=/data/SimLibriCSS/$name/wav
   # List the source files.
   datalist=$tgtroot/$name.list
   python source_md/gen_filelist.py --srcdir $srcdir --outlist $datalist
   splitdir=${tgtroot}/split${nj}
    mkdir -p ${splitdir}/log

   # Create a JSON file for the source data set. (~10 min with nj=32)
   datajson=$tgtroot/${name}.json
   ${gen_cmd} JOB=1:${nj} ${splitdir}/log/list2json.JOB.log \
        python  source_md/list2json_librispeech.py --novad --input_list ${splitdir}/${name}.JOB.list --output_file ${splitdir}/${name}.JOB.json
   

   python source_md/mergejsons.py $(for j in $(seq ${nj}); do echo ${splitdir}/${name}.${j}.json; done) > $datajson

  done

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
  #echo "Generate mixture specs, its output is only mixspec.json"
  gen_cmd=source_md/run.pl
  dyncfg=source_md/meeting_dynamics.json # it is from https://github.com/shanguanma/jsalt2020_simulate/blob/master/configs/common/meeting_dynamics.json
  roomcfg=source_md/meeting_reverb.json # it is from https://github.com/shanguanma/jsalt2020_simulate/blob/master/configs/common/meeting_reverb.json

  for name  in train dev test; do
   tgtroot=/data/SimLibriCSS/SimLibriCSS-$name
   srcdir=/data/SimLibriCSS/$name/wav

   # Create a JSON file for the source data set. (~10 min with nj=32)
   datajson=$tgtroot/${name}.json

   # Generate mixture specs. 
   tgtdir=$tgtroot/wav
   specjson=$tgtroot/mixspec.json
   splitdir=${tgtroot}/split${nj}
   ##Generate mixture specs, its output is only mixspec.json
   python  source_md/gen_mixspec_mtg.py --inputfile $datajson --outputfile $specjson --targetdir $tgtdir --random_seed 0 --config $dyncfg



   # Split $tgtroot/mixspec.json into several smaller json files: $splitdir/mixspec.JOB.json
   python source_md/splitjson.py --inputfile $specjson --number_splits $nj --outputdir $splitdir

   # Generate mixed audio files and store at $tgtdir
   mixlog=$tgtroot/mixlog.json
   opts=''
   ${gen_cmd} JOB=1:${nj} ${splitdir}/log/mixlog.JOB.log \
    python source_md/mixaudio_mtg.py $opts --iolist ${splitdir}/mixspec.JOB.json \
            --cancel_dcoffset --random_seed JOB --sample_rate 16000 \
	    --log ${splitdir}/mixlog.JOB.json --mixers_configfile $roomcfg
   
   python source_md/mergejsons.py $(for j in $(seq ${nj}); do echo ${splitdir}/mixlog.${j}.json; done) > $mixlog

   ## note: 
   ## mixer audios are stored at $tgtdir, their details are stored at mixlog.json
   ## mixer audio contains reverb and overlap multi speakers.
   ## At this point, the simulation processing process end.
done
fi


## now, I will prepare superviser diarization data format using the above simulated data.
##  these data will be used to train ts_vad diarization model.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
  echo "convert json into rttm" 
  for name  in train dev test; do
   input_file=/data/SimLibriCSS/SimLibriCSS-$name/mixspec.json
   output_file=/data/SimLibriCSS/SimLibriCSS-$name/rttm
   python source_md/convert_simulate_libricss_json_to_rttm.py\
	  --mixspec_json $input_file\
	 --output_rttm $output_file
  done

fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then

   echo "prepare target label json file"
   for name  in train dev test; do
     rttm_file=/data/SimLibriCSS/SimLibriCSS-$name/rttm
     input_wav_dir=/data/SimLibriCSS/SimLibriCSS-$name/wav
     output_dir=/data/SimLibriCSS/SimLibriCSS-$name
     python  source_md/prepare_simulate_libricss.py \
              --path_rttm $rttm_file \
	      --path_wav $input_wav_dir\
	      --out_dir $output_dir \
              --label_rate 25\
	      --type $name
   done

fi
