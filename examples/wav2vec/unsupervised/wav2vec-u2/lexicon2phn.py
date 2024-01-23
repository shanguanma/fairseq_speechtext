#!/usr/bin/env python

import sys
import logging


if __name__ == "__main__":
   formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

   logging.basicConfig(format=formatter, level=logging.INFO)
   phoneset=list()
   for line in sys.stdin:
       line = line.strip().split()
       for phone in line[1:]:
           if phone not in phoneset:
               phoneset.append(phone)
   
   for i in sorted(phoneset):
       print(f"{i}")

