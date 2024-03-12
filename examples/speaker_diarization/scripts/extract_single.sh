librimix_path=/mnt/bn/junyi-nas2/librimix/Libri2Mix/wav16k/max/test
librispeech_path=/mnt/bn/junyi-nas2/librimix/LibriSpeech/test-clean
model_path=/mnt/bn/junyi-nas2/codebase/joint-optimization/exp_fairseq/joint/joint_aftertsvad_combine160_usevloss_maxmode/checkpoints/checkpoint_10_100000.pt

mix_path=${librimix_path}/mix_both/1089-134686-0000_121-127105-0031.wav
ref_path=${librispeech_path}/1089/134686/1089-134686-0031.flac,${librispeech_path}/121/127105/121-127105-0005.flac

python3 ts_vad/generate_spex_joint_sutt.py --mixture_path ${mix_path} --ref_path ${ref_path} --model_path ${model_path}