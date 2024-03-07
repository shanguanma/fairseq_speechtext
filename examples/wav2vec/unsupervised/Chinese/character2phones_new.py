#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import sys
import codecs
import logging
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    lexicon = sys.argv[1]
    scp_text = sys.argv[2]

    wrd_to_phn = {}

    with open(lexicon, "r") as lf:
        for line in lf:
            items = line.rstrip().split()
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1:]
    #print(f"{wrd_to_phn}")

    with open(scp_text, 'r')as f, open(sys.argv[3],'w')as fw, open(sys.argv[4],'w') as fd:
        for line in f:
            line = line.strip().split()
            words = line[1:]
            key = line[0]
            new_line = key + '\t'
            for w in words:
                if w not in wrd_to_phn:
                    logging.info(f"word not in lexicon: {w}")
            if not all(w in wrd_to_phn for w in words):
                logging.info(f"this line ({line}) skip!!")
                fd.write(f"{key}\n")
                continue
            
            for i, w in enumerate(words):
                new_line += ' '.join(wrd_to_phn[w])
                new_line += ' '
            new_line = new_line.replace('  ', ' ')
            fw.write(f"{new_line}\n")



"""
    with open(scp_text, 'r')as f:
        with open(sys.argv[3],'w')as fw:
            for line in f:
                words = line.rstrip().split()
                new_line = words[0] + '\t' 
                for word in words[1:]:
                    if word in wrd_to_phn.keys():
                        new_line += ' '.join(wrd_to_phn[word])
                        new_line += ' '
                new_line = new_line.replace('  ', ' ')
                fw.write(f"{new_line}\n")
"""
