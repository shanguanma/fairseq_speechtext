#!/usr/bin/env python
"""
using silero-vad package to remove silence, it will faster than rVADfast
"""



import os
import argparse
import torch
import torchaudio
import logging
#import tqdm
from tqdm import tqdm
import soundfile
import torchaudio.backend.sox_io_backend as sox
from typing import List
import time
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
"""
torchrun --nproc_per_node=5 --master_port=12345  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/silero-vad.py --wavscp tests/wav20.scp  --out tests/wenetspeech_wo_silence_silero_vad
"""


from utils_vad import (init_jit_model,
                       get_speech_timestamps,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       OnnxWrapper)



def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def versiontuple(v):
    splitted = v.split('+')[0].split(".")
    version_list = []
    for i in splitted:
        try:
            version_list.append(int(i))
        except:
            version_list.append(0)
    return tuple(version_list)

def silero_vad(model_dir, onnx=False, force_onnx_cpu=False):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """

    if not onnx:
        installed_version = torch.__version__
        supported_version = '1.12.0'
        if versiontuple(installed_version) < versiontuple(supported_version):
            raise Exception(f'Please install torch {supported_version} or greater ({installed_version} installed)')

    #model_dir = os.path.join(os.path.dirname(__file__), 'files')
    if onnx:
        model = OnnxWrapper(os.path.join(model_dir, 'silero_vad.onnx'), force_onnx_cpu)
    else:
        model = init_jit_model(os.path.join(model_dir, 'silero_vad.jit'))
    utils = (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks)

    return model, utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavscp", default="", type=str)
    parser.add_argument("--onnx", type=str2bool, default=True,help="if true, it will vad onnx model, it will faster than vad jit model")
    parser.add_argument("--out", type=str)
    parser.add_argument('--segments', default=None, help='segments file')
    parser.add_argument('--text_file', help='text file')
    parser.add_argument('--resample',
                        type=int,
                        default=16000,
                        help='segments file')
    return parser



def load_data_list_ref(wavscp) -> List[tuple[str,str]]:
    # load paths
    #paths = dict()
    paths = []
    with open(wavscp) as f:
        #root = next(f).rstrip()
        #for line in f:
        #    paths.append(os.path.join(root, line.rstrip().split("\t")[0]))
        for line in f:
            key, path = line.rstrip().split(maxsplit=1)
            #paths[key] = path
            paths.append((key,path))
            #paths.append(line.rstrip().split()[1])
    return paths


def load_audio_ref(audio) -> torch.float32:
    ## soundfile read audio, get float64 data, however torchaudio.load get float32 data.
    data, sr = soundfile.read(audio) # data is np.float64, it is same as  stage of computing vads.
    data = torch.from_numpy(data).to(torch.float32) # data is torch.float32 
    assert sr==16000,f"expected sample rate is 16000, however uttid is {uttid}, sample rate: {sr}"
    return data

def write_audio(path: str, data: torch.Tensor, sampling_rate: int=16000):
    if data is None:
        logging.info(f"remove silence, audio is empty!!!")
    else:
        if torch.is_tensor(data):
            data = data.numpy()
        soundfile.write(path,data,samplerate=sampling_rate, format='OGG', subtype='OPUS')


        
def creat_output_wavname(args,path)->str:
    ## assume the wavform path is as follows:
    ## '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus'
    # >>> os.path.dirname(path).split("/",maxsplit=6)
    # ['', 'mntcephfs', 'lee_dataset', 'asr', 'WenetSpeech', 'untar', 'audio/train/youtube/B00000']
    # >>> os.path.splitext(path)    
    # ('/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84', '.opus')
    # >>> os.path.splitext(path)[0].split("/",maxsplit=6) 
    # ['', 'mntcephfs', 'lee_dataset', 'asr', 'WenetSpeech', 'untar', 'audio/train/youtube/B00000/Y0000000000_--5llN02F84']
    os.makedirs(args.out,exist_ok=True)            
    out = os.path.splitext(path)[0].split("/",maxsplit=6)[-1] 
    #outpath = args.out + '/'+ out + ".wav" ## torchaudio.save don't support save as opus.
    outpath = args.out + '/'+ out + ".opus" # soundfile support save as opus, reference:https://github.com/bastibe/python-soundfile/issues/252  
    output_dir = args.out + '/' + os.path.dirname(path).split("/",maxsplit=6)[-1]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir,exist_ok=True)   
    return outpath



def load_data_list(args):

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    no_segments = True
    segments_table = {}
    if args.segments is not None:
        no_segments = False
        with open(args.segments, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 4
                segments_table[arr[0]] = (arr[1], float(arr[2]), float(arr[3]))

    data = []
    with open(args.text_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ''
            if no_segments:
                assert key in wav_table
                wav = wav_table[key]
                data.append((key, txt, wav))
            else:
                wav_key, start, end = segments_table[key]
                wav = wav_table[wav_key]
                data.append((key, txt, wav, start, end))
    return data,no_segments

def load_audio(item, no_segments,resample=16000)-> torch.float32:
    #for item in data_list:
    if no_segments:
        key, txt, wav = item
    else:
        key, txt, wav, start, end = item    
    suffix = wav.split('.')[-1]
    assert suffix in AUDIO_FORMAT_SETS
    if no_segments:
        # read & resample
        #ts = time.time()
        audio, sample_rate = sox.load(wav, normalize=False)
        if sample_rate != resample:
            audio = torchaudio.transforms.Resample(
                sample_rate, resample)(audio.float())
    else:
        waveforms, sample_rate = sox.load(wav, normalize=False)
        start = int(start * sample_rate)
        end = int(end * sample_rate)
        audio = waveforms[:1, start:end]
        # resample
        if sample_rate != resample:
            if not audio.is_floating_point():
                # normalize the audio before resample
                # because resample can't process int audio
                audio = audio / (1 << 15)
                audio = torchaudio.transforms.Resample(
                    sample_rate, resample)(audio)
                audio = (audio * (1 << 15)).short()
            else:
                audio = torchaudio.transforms.Resample(
                    sample_rate, resample)(audio)
    assert audio.dtype==torch.float32,f"audio.dtype: {audio.dtype}"
    return audio

from torch.distributed.elastic.multiprocessing.errors import record
@record
def main():
    parser = get_parser()
    args = parser.parse_args()
     

    ## prepared mulit-process utils
    rank = int(os.environ['LOCAL_RANK'])        ## processing id
    threads_num = int(os.environ['WORLD_SIZE']) ## cpu numbers, is setted by --nproc_per_node 
    logging.info("rank {}/{}.".format(
        rank, threads_num,
    ))




    ## load vad model
    #logging.info(f"local_all_paths: {local_all_paths}")
    local_model_path="/mntnfs/lee_data1/maduo/codebase/silero-vad/files"
    model, utils = silero_vad(model_dir=local_model_path, onnx=args.onnx, force_onnx_cpu=False)
    (get_speech_timestamps,
    _,
    _,
    _,
    collect_chunks) = utils



    ## split data on rank
    data_list, no_segments = load_data_list(args)
    #paths.sort(key=lambda x: x[0])
    local_data_list = data_list[rank::threads_num] 

    i=0
    #for i, (uttid, path) in tqdm(enumerate(local_all_paths), total=len(local_all_paths),ascii=True):
    for i, item in tqdm(enumerate(local_data_list),total=len(local_data_list),ascii=True):
        audio = load_audio(item,no_segments, args.resample) # load audio and resample
        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=args.resample)
        logging.info(f"speech_timestamps: {speech_timestamps}")
        if speech_timestamps is  not None:
            data_wo_silence = collect_chunks(speech_timestamps, data) ## data_wo_silence maybe None
            if data_wo_silence is None:
                continue
            outpath = creat_output_wavname(args,path)
            write_audio(outpath, data_wo_silence, sampling_rate=args.resample)
            i = i+1
            if i%100==0:
                logging.info("{}/{}: process {}.".format(rank, threads_num, uttid))
    logging.info(f"write {i} utterances , finish!!!!!")

if __name__ == "__main__":
    
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

    """
    onnx=True, I use torchrun multi processing 
    it consums about 9mins processing 20 utterances and write without silence audio.
     torchrun --nproc_per_node=5 --master_port=12345  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/silero-vad.py --wavscp tests/wav20.scp --onnx true --out tests/wenetspeech_wo_silence_silero_vad
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
2024-01-11 17:06:21,805 INFO [silero-vad.py:133] rank 0/5.
2024-01-11 17:06:21,806 INFO [silero-vad.py:140] local_all_paths: [('Y0000000000_--5llN02F84', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus'), ('Y0000000006_-0IBbmCokvY', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000006_-0IBbmCokvY.opus'), ('Y0000000011_-1800ATDh64', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000011_-1800ATDh64.opus'), ('Y0000000016_-1f3NNSInzU', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000016_-1f3NNSInzU.opus')]
2024-01-11 17:06:21,827 INFO [silero-vad.py:133] rank 3/5.
2024-01-11 17:06:21,827 INFO [silero-vad.py:140] local_all_paths: [('Y0000000003_--tTfmNRKHU', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000003_--tTfmNRKHU.opus'), ('Y0000000009_-0p8pYdlfjY', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000009_-0p8pYdlfjY.opus'), ('Y0000000014_-1_DSSMnDCE', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000014_-1_DSSMnDCE.opus'), ('Y0000000019_-1rYBgd8XXs', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000019_-1rYBgd8XXs.opus')]
2024-01-11 17:06:21,828 INFO [silero-vad.py:133] rank 4/5.
2024-01-11 17:06:21,829 INFO [silero-vad.py:140] local_all_paths: [('Y0000000005_-0CH_WrGBKY', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000005_-0CH_WrGBKY.opus'), ('Y0000000010_-0zkH4K38Z0', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000010_-0zkH4K38Z0.opus'), ('Y0000000015_-1b96ICVDV4', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000015_-1b96ICVDV4.opus'), ('Y0000000020_-1rpxhaKHHg', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000020_-1rpxhaKHHg.opus')]
2024-01-11 17:06:21,872 INFO [silero-vad.py:133] rank 1/5.
2024-01-11 17:06:21,872 INFO [silero-vad.py:133] rank 2/5.
2024-01-11 17:06:21,872 INFO [silero-vad.py:140] local_all_paths: [('Y0000000001_--TiI-YRU38', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000001_--TiI-YRU38.opus'), ('Y0000000007_-0P6CYbKnsQ', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000007_-0P6CYbKnsQ.opus'), ('Y0000000012_-1EdlcTjw8k', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000012_-1EdlcTjw8k.opus'), ('Y0000000017_-1fY8qw-VrM', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000017_-1fY8qw-VrM.opus')]
2024-01-11 17:06:21,873 INFO [silero-vad.py:140] local_all_paths: [('Y0000000002_--s1SMM6PBU', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000002_--s1SMM6PBU.opus'), ('Y0000000008_-0RZxs2c3X8', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000008_-0RZxs2c3X8.opus'), ('Y0000000013_-1T8BwkquEI', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000013_-1T8BwkquEI.opus'), ('Y0000000018_-1pX3Y3M0GI', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000018_-1pX3Y3M0GI.opus')]
2024-01-11 17:06:21.950344121 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '131'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950345785 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '131'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950374782 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '136'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950377848 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '136'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950382364 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '139'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950388022 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '139'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950391250 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '140'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950398665 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '140'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950401486 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '134'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950408693 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '134'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950459025 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '131'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950478236 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '136'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950484987 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '139'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950491954 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '140'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950497378 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '628'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950498511 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '134'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950504757 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '623'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950511317 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '629'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950510863 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '628'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950517099 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '620'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950521336 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '623'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950524310 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '625'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950528871 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '629'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950537215 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '620'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950528149 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '131'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950544544 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '625'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950550238 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '136'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950557039 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '139'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950563662 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '140'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950570393 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '134'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950594728 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '628'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950601572 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '623'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950607705 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '629'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950614032 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '620'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950620691 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '625'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950664390 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '628'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950671405 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '623'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950677798 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '629'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950683689 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '620'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.950689982 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '625'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956683553 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '131'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956722559 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '136'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956732660 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '139'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956743392 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '140'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956754745 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '134'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956890658 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '628'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956901990 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '623'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956912552 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '629'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956922371 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '620'. It is not used by any node and should be removed from the model.
2024-01-11 17:06:21.956932993 [W:onnxruntime:, graph.cc:3553 CleanUnusedInitializersAndNodeArgs] Removing initializer '625'. It is not used by any node and should be removed from the model.
100%|###############################################################################################################################################################| 4/4 [06:03<00:00, 90.85s/it]
2024-01-11 17:12:25,448 INFO [silero-vad.py:164] write 4 utterances , finish!!!!!
100%|###############################################################################################################################################################| 4/4 [06:24<00:00, 96.18s/it]
2024-01-11 17:12:46,773 INFO [silero-vad.py:164] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [06:51<00:00, 102.99s/it]
2024-01-11 17:13:14,010 INFO [silero-vad.py:164] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [07:11<00:00, 107.87s/it]
2024-01-11 17:13:33,543 INFO [silero-vad.py:164] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [08:44<00:00, 131.22s/it]
2024-01-11 17:15:06,933 INFO [silero-vad.py:164] write 4 utterances , finish!!!!! 
    """


    """
    it consums about 16mins processing 20 utterances and write without silence audio.
     torchrun --nproc_per_node=5 --master_port=12345  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/silero-vad.py --wavscp tests/wav20.scp --onnx false --out tests/wenetspeech_wo_silence_silero_vad_jit
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
2024-01-11 17:23:54,517 INFO [silero-vad.py:152] rank 0/5.
2024-01-11 17:23:54,518 INFO [silero-vad.py:159] local_all_paths: [('Y0000000000_--5llN02F84', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus'), ('Y0000000006_-0IBbmCokvY', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000006_-0IBbmCokvY.opus'), ('Y0000000011_-1800ATDh64', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000011_-1800ATDh64.opus'), ('Y0000000016_-1f3NNSInzU', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000016_-1f3NNSInzU.opus')]
2024-01-11 17:23:54,543 INFO [silero-vad.py:152] rank 2/5.
2024-01-11 17:23:54,544 INFO [silero-vad.py:159] local_all_paths: [('Y0000000002_--s1SMM6PBU', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000002_--s1SMM6PBU.opus'), ('Y0000000008_-0RZxs2c3X8', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000008_-0RZxs2c3X8.opus'), ('Y0000000013_-1T8BwkquEI', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000013_-1T8BwkquEI.opus'), ('Y0000000018_-1pX3Y3M0GI', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000018_-1pX3Y3M0GI.opus')]
2024-01-11 17:23:54,575 INFO [silero-vad.py:152] rank 4/5.
2024-01-11 17:23:54,576 INFO [silero-vad.py:159] local_all_paths: [('Y0000000005_-0CH_WrGBKY', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000005_-0CH_WrGBKY.opus'), ('Y0000000010_-0zkH4K38Z0', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000010_-0zkH4K38Z0.opus'), ('Y0000000015_-1b96ICVDV4', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000015_-1b96ICVDV4.opus'), ('Y0000000020_-1rpxhaKHHg', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000020_-1rpxhaKHHg.opus')]
2024-01-11 17:23:54,598 INFO [silero-vad.py:152] rank 3/5.
2024-01-11 17:23:54,599 INFO [silero-vad.py:159] local_all_paths: [('Y0000000003_--tTfmNRKHU', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000003_--tTfmNRKHU.opus'), ('Y0000000009_-0p8pYdlfjY', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000009_-0p8pYdlfjY.opus'), ('Y0000000014_-1_DSSMnDCE', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000014_-1_DSSMnDCE.opus'), ('Y0000000019_-1rYBgd8XXs', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000019_-1rYBgd8XXs.opus')]
2024-01-11 17:23:54,643 INFO [silero-vad.py:152] rank 1/5.
2024-01-11 17:23:54,644 INFO [silero-vad.py:159] local_all_paths: [('Y0000000001_--TiI-YRU38', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000001_--TiI-YRU38.opus'), ('Y0000000007_-0P6CYbKnsQ', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000007_-0P6CYbKnsQ.opus'), ('Y0000000012_-1EdlcTjw8k', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000012_-1EdlcTjw8k.opus'), ('Y0000000017_-1fY8qw-VrM', '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000017_-1fY8qw-VrM.opus')]
100%|##############################################################################################################################################################| 4/4 [11:48<00:00, 177.18s/it]
2024-01-11 17:35:43,551 INFO [silero-vad.py:183] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [12:50<00:00, 192.72s/it]
2024-01-11 17:36:45,675 INFO [silero-vad.py:183] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [13:24<00:00, 201.10s/it]
2024-01-11 17:37:19,212 INFO [silero-vad.py:183] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [13:32<00:00, 203.15s/it]
2024-01-11 17:37:27,392 INFO [silero-vad.py:183] write 4 utterances , finish!!!!!
100%|##############################################################################################################################################################| 4/4 [16:05<00:00, 241.27s/it]
2024-01-11 17:39:59,880 INFO [silero-vad.py:183] write 4 utterances , finish!!!!!
    """
