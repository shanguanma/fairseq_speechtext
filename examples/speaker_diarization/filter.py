#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
import sys


if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    with open(input, 'r')as f,open(output,'w')as fw:
        for line in f:
            line = line.strip().split()
            if line[7]!="G00000000":
                print(" ".join(line), file=fw)
