#!/usr/bin/env python
import torch

import sys

def count_frames(speechcode):
    speechlist=[]
    with open(speechcode,'r')as f:
       for line in f:
           line = line.strip().split()
           #print(len(line))
           line = [int(s) for s in line]
           line_t = torch.tensor(line)
           line_unique, count = torch.unique_consecutive(line_t, return_counts=True)
           #print(line_t.shape)
           #print(line_unique, count)

           ele2c=dict() ## phone to its frames of list
           #clist= []
           line_unique_list = line_unique.tolist()
           count_list = count.tolist()
           for ele,c in zip(line_unique_list, count_list):
               #clist.append((ele,c))
               ele = str(ele)
               c = str(c)
               if ele not in ele2c.keys():
               #
                   ele2c[ele] = [c]
               else:

                   ele2c[ele] += [c] ### list splicing
           #print(clist)
           print(ele2c)
           speechlist.append(ele2c)
    return speechlist ## dict of list, every dict is phone and its frames count of a utterance 



if __name__ == "__main__":
   speechcode=sys.argv[1]
   textcode=sys.argv[2]
   speechlist = count_frames(speechcode)
   textlist=[]

   with open(textcode, 'r') as f:
       for line in f:
           line = line.strip().split()
           textlist.append(line)
   print(f"textlist: {textlist}")
   import secrets
   import random
   
   speech_count_dict = secrets.choice(speechlist) #Choose a random item from the list securely
   print(f"speech_count_dict: {speech_count_dict}")
   phns = len(speech_count_dict)
   k = phns//4
   textutt = secrets.choice(textlist)
   print(f"textutt: {textutt}")
   phncode_keys_list =random.choices(list(speech_count_dict),k=k)
   print(f"phncode_keys_list: {phncode_keys_list}")
   new_l = []
   for code in textutt:
       #code = int(code)
       if code in phncode_keys_list:
           print(f"code: {code}")
           frames_count_list = speech_count_dict[code]
           print(f"frames_count_list: {frames_count_list}")
           n = secrets.choice(frames_count_list)
           print(f"n: {n}")
           new_l.extend([code] * int(n))
       else:
           new_l.extend([code])
   new_text = ' '.join(new_l)
   print(new_text)

