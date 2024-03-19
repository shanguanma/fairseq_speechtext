#!/usr/bin/env python

import sys


if __name__== "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    phones_set=[]
    with open(input, 'r')as f:
        for line in f:
            line = line.strip().split()
            for i in line:
                if i not in phones_set:
                    phones_set.append(i)

    last_id = len(phones_set)
    print(last_id)
    with open(output, 'w')as fo:
        for i, phone in enumerate(phones_set):
            print(i, phone)
            fo.write(f"{phone} {i}\n")
        fo.write(f"<SIL> {last_id}\n") 
    
