#!/usr/bin/env python
"""
"""



import os
import argparse
import torch
import torchaudio
import logging
#import tqdm
from tqdm import tqdm
import soundfile
#import torchaudio.backend.sox_io_backend as sox
from typing import List
import time
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


"""
torchrun --nproc_per_node=5 --master_port=12345  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/format_wav_scp.py --wav_file tests/test_kaldi_format/wav.scp  --out tests/test_kaldi_format/no_segements --segments tests/test_kaldi_format/segments --text_file tests/test_kaldi_format/text
"""



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

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_file", default="", type=str)
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


        
def creat_output_wavname_ref(args,path)->str:
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



def creat_output_wavname(args,path, key, no_segments)->str:
    if no_segments:
        os.makedirs(args.out,exist_ok=True)
        out = os.path.splitext(path)[0].split("/",maxsplit=6)[-1]
        #outpath = args.out + '/'+ out + ".wav" ## torchaudio.save don't support save as opus.
        outpath = args.out + '/'+ out + ".opus" # soundfile support save as opus, reference:https://github.com/bastibe/python-soundfile/issues/252
        output_dir = args.out + '/' + os.path.dirname(path).split("/",maxsplit=6)[-1]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir,exist_ok=True)
        return outpath
    else:
        os.makedirs(args.out,exist_ok=True)
        out = os.path.dirname(path).split("/",maxsplit=6)[-1]
        output_dir=args.out + '/'+ out
        os.makedirs(output_dir,exist_ok=True)
        outpath = output_dir + '/'+ key + ".opus"
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
        #audio, sample_rate = torchaudio.load(wav, normalize=False,backend="soundfile")
        data, sample_rate = soundfile.read(wav) # data is np.float64, it is same as  stage of computing vads.
        data = torch.from_numpy(data).to(torch.float32) # data is torch.float32
        #assert sr==16000,f"expected sample rate is 16000, however uttid is {uttid}, sample rate: {sr}"
        if sample_rate != resample:
            audio = torchaudio.transforms.Resample(
                sample_rate, resample)(audio.float())
    else:
        #waveforms, sample_rate = torchaudio.load(wav, normalize=False,backend="soundfile")
        waveforms, sample_rate = soundfile.read(wav) # data is np.float64, it is same as  stage of computing vads.
        waveforms = torch.from_numpy(waveforms).to(torch.float32) # data is torch.float32
        #assert sr==16000,f"expected sample rate is 16000, however uttid is {uttid}, sample rate: {sr}"
        start = int(start * sample_rate)
        end = int(end * sample_rate)
        audio = waveforms[start:end]
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
    return audio, wav, key





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






    ## split data on rank
    data_list, no_segments = load_data_list(args)
    #paths.sort(key=lambda x: x[0])
    local_data_list = data_list[rank::threads_num] 

    i=0
    #for i, (uttid, path) in tqdm(enumerate(local_all_paths), total=len(local_all_paths),ascii=True):
    for i, item in tqdm(enumerate(local_data_list),total=len(local_data_list),ascii=True):
        audio, path, key = load_audio(item,no_segments, args.resample) # load audio and resample
        outpath = creat_output_wavname(args, path, key, no_segments)
        write_audio(outpath, audio, sampling_rate=args.resample)
        i = i+1
        if i%100==0:
            #logging.info("{}/{}: process {}.".format(rank, threads_num, uttid))
            logging.info("{}/{}: process {}.".format(rank, threads_num, key))
    logging.info(f"write {i} utterances , finish!!!!!")

if __name__ == "__main__":
    
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

