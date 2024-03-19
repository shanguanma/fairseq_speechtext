#!/usr/bin/env python

import sys

if __name__ == '__main__':
    file = sys.argv[1]
    output = sys.argv[2]
    words=[]
    with open(file, 'r') as fi:
        for line in fi:
            line = line.strip().split()
            #print(line)
            for i in line:
                if i not in words:
                    words.append(i)
    #print()    
    with open(output,'w')as fo:
        for j in words:
            fo.write(f"{j}\n")
        
