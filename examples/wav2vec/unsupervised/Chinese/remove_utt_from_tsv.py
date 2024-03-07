#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import sys
import codecs
import logging
import os
def get_fname(line):
    # line : base_on_silero-vad_onnx_torchrun_parallel/audio/dev/third_party/B00000/DEV_T0000000000_S00000.opus      65344
    p = os.path.basename(line.split("\t")[0]) # DEV_T0000000000_S00000.opus
    p = os.path.splitext(p)[0] # DEV_T0000000000_S00000
    return p
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    #remove_utt = sys.argv[1]
    tsv_file = sys.argv[2]
    text_file = sys.argv[3]
    new_tsv_file = sys.argv[4]
    new_text_file = sys.argv[5]
    remove_utt = []
    with open(sys.argv[1],'r')as f:
        for line in f:
            line = line.strip()
            remove_utt.append(line)

   
    with open(tsv_file,'r')as fr,open(new_tsv_file,'w')as ftsv:
        root = next(fr).rstrip()
        ftsv.write(f"{root}\n")
        for line in fr:
            line = line.strip()
            fname = get_fname(line)
            if fname in remove_utt: 
                continue # remove utt
            #print(f"{line}")
            ftsv.write(f"{line}\n")

    with open(text_file,'r')as fr,open(new_text_file,'w')as ftext:
        for line in fr:
            line_split = line.strip().split()
            key = line_split[0]
            if key in remove_utt:
                continue # remove utt
            ftext.write(f"{line}")




