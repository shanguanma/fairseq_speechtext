#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import sys
import codecs

if __name__ == "__main__":
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

