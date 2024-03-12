from glob import glob
from tqdm import tqdm
import os
import scipy.signal
import soundfile as sf
from joblib import Parallel, delayed

audio_dir = '/workspace2/junyi/datasets/callhome_sim/--no-use-rirs--use-noises/validation/wavs'
new_audio_dir = '/workspace2/junyi/datasets/callhome_sim/--no-use-rirs--use-noises/validation/wavs_16k'

os.makedirs(new_audio_dir, exist_ok=True)

def convert(audio_path):
    print(audio_path)
    audio_name = audio_path.strip('\n').split('/')[-1]
    audio, sr = sf.read(audio_path)           
    audio_re = scipy.signal.resample(audio, int((16000 * len(audio)) / 8000)) # upsample to 16000
    # import pdb; pdb.set_trace()
    sf.write(f"{new_audio_dir}/{audio_name}", audio_re, 16000)

Parallel(n_jobs=80)(delayed(convert)(audio_path) for audio_path in glob(audio_dir + '/*.wav'))
