#!/usr/bin/env python3


import os
import sys
import textgrid

if __name__ == "__main__":
    input=sys.argv[1] 
    #output=sys.argv[2]
    # Read a TextGrid object from a file.
    tg = textgrid.TextGrid.fromFile(input)
    # Read a PointTier object.
    print("------- PointTier Example -------")
    print(tg[1])
    print(tg[1][0])
    print(tg[1][0].text)
    #print(tg[1][0].time)
    #print(tg[1][0].mark)        
