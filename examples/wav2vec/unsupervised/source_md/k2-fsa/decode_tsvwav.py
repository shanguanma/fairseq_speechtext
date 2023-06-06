#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-ncnn Python API to recognize
a single file.

Please refer to
https://k2-fsa.github.io/sherpa/ncnn/index.html
to install sherpa-ncnn and to download the pre-trained models
used in this file.
"""

import time
import wave
import soundfile 
import numpy as np
import sherpa_ncnn


def main():
    # Please refer to https://k2-fsa.github.io/sherpa/ncnn/index.html
    # to download the model files
    model_hubs="./model_hub/k2-fsa/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/"
    tokens=model_hubs+"/tokens.txt"
    encoder_param=model_hubs+"/encoder_jit_trace-pnnx.ncnn.param"
    encoder_bin=model_hubs+"/encoder_jit_trace-pnnx.ncnn.bin"
    decoder_param=model_hubs+"/decoder_jit_trace-pnnx.ncnn.param"
    decoder_bin=model_hubs+"/decoder_jit_trace-pnnx.ncnn.bin"
    joiner_param=model_hubs+"/joiner_jit_trace-pnnx.ncnn.param"
    joiner_bin=model_hubs+"/joiner_jit_trace-pnnx.ncnn.bin"
    num_threads=4
    recognizer = sherpa_ncnn.Recognizer(
        tokens=tokens,
        encoder_param=encoder_param,
        encoder_bin=encoder_bin,
        decoder_param=decoder_param,
        decoder_bin=decoder_bin,
        joiner_param=joiner_param,
        joiner_bin=joiner_bin,
        num_threads=num_threads,
    )
    #tsvfiles="./tests/dev-clean10.tsv"
    tsvfiles="./source_md/k2-fsa/tests/wavtsv.tsv"
    #filename = model_hubs + "/test_wavs/1.wav"
    with open(tsvfiles,"r") as f:
        root_dir=''
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
               root_dir = line
            else:
                line = line.split()[0]
                filename = root_dir+ '/' + line
                print(f"filename: {filename}")                        
                with wave.open(filename) as f:
                #with soundfile.SoundFile(filename) as f:
                    # Note: If wave_file_sample_rate is different from
                    # recognizer.sample_rate, we will do resampling inside sherpa-ncnn
#                    wave_file_sample_rate = f.getframerate()
#                    num_channels = f.getnchannels()
#                    assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
#                    f.seek(start)
#                    if end is not None:
#                        frames = end - start
#                    else:
#                        frames = -1
#                    array = f.read(f.frames)
#                    samples = array
#                    num_channels = f.channels
#                    wave_file_sample_rate=f.samplerate
#                    #num_samples = f.getnframes()
#                    #samples = f.readframes(num_samples)
#                    samples_int16 = np.frombuffer(samples, dtype=np.int16)
#                    samples_int16 = samples_int16.reshape(-1, num_channels)[:, 0]
#                    samples_float32 = samples_int16.astype(np.float32)
#
#                    samples_float32 = samples_float32 / 32768
                
                    wave_file_sample_rate = f.getframerate()
                    num_channels = f.getnchannels()
                    assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
                    num_samples = f.getnframes()
                    samples = f.readframes(num_samples)
                    samples_int16 = np.frombuffer(samples, dtype=np.int16)
                    samples_int16 = samples_int16.reshape(-1, num_channels)[:, 0]
                    samples_float32 = samples_int16.astype(np.float32)

                    samples_float32 = samples_float32 / 32768
                # simulate streaming
                chunk_size = int(0.1 * wave_file_sample_rate)  # 0.1 seconds
                start = 0
                while start < samples_float32.shape[0]:
                    end = start + chunk_size
                    end = min(end, samples_float32.shape[0])
                    recognizer.accept_waveform(wave_file_sample_rate, samples_float32[start:end])
                    start = end
                    text = recognizer.text
                    if text:
                        print(text)

                    # simulate streaming by sleeping
                    time.sleep(0.1)

                tail_paddings = np.zeros(int(wave_file_sample_rate * 0.5), dtype=np.float32)
                recognizer.accept_waveform(wave_file_sample_rate, tail_paddings)
                recognizer.input_finished()
                text = recognizer.text
                if text:
                    print(text)


if __name__ == "__main__":
    main()
