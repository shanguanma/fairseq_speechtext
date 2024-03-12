import os
from glob import glob
import soundfile

skip_wav_path = "/mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-test/skip_wavs.txt"
librispeech = "/mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/librispeech/LibriSpeech"

subset = ["test-clean", "test-other"]

with open(skip_wav_path) as f:
    for line in f:
        line = line.strip()
        spk_id = line.split("/")[-1][:-4]
        for ss in subset:
            target_path = f"{librispeech}/{ss}/{spk_id}"
            if os.path.exists(target_path):
                print(target_path)
                new_target_audio = []
                for wav in glob(f"{target_path}/*/*.flac"):
                    # import pdb; pdb.set_trace()
                    if len(new_target_audio) > 30 * 16000:
                        break
                    new_target_audio.extend(soundfile.read(wav)[0])
                break

        # os.rename(line, line[:-4] + '_bku.wav')
        soundfile.write(line, new_target_audio, 16000)
