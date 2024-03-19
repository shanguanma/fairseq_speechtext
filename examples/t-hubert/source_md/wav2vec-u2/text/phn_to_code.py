#!/usr/bin/env python3


import sys
import os

if __name__ == "__main__":
    input_text=sys.argv[1]
    dict_file=sys.argv[2]
    output=sys.argv[3]
    dict1={}
    with open(dict_file, 'r')as f:
        for line in f:
            line = line.strip()
            line = line.split()
            dict1[line[0]] = line[1]
    

    with open(input_text, 'r')as f, open(output, 'w')as fw:
         for line in f:
             line = line.strip().split()
             #print(f"line: {line}")
             utterance=[]
             for i in line:
                 utterance.append(dict1[i])
             print(" ".join(utterance), file=fw)         
