#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import sys
import os 
if __name__ == "__main__":
    tsv_file = sys.argv[1]
    scp_file = sys.argv[2]

    wavnames1=[]
    with open(scp_file,'r')as f:
        for line in f:
            line = line.strip().split()
            wavnames1.append(line[0])


    wavnames2=[]
    with open(tsv_file,'r')as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        for line in lines:
            if len(line)>0:
                path = line.split("\t")[0]
                waename =  os.path.basename(path).split(".")[0]
                wavnames2.append(waename) 

    miss_file_names=list(set(wavnames1).difference(set(wavnames2)))
    print(f"miss_files: {list(set(wavnames1).difference(set(wavnames2)))}")    # 使用 difference 求a与b的差(补)集：求b中有而a中没有的元素，输出： [5, 6, 7, 8, 9]
    #print(list(set(a).difference(set(b))) )   # 使用 difference 求a与b的差(补)集：求a中有而b中没有的元素，输出：[5, 6, 7, 8, 9]
    with open(sys.argv[3],'w')as fw:
        with open(scp_file,'r')as f:
            for line in f:
                line = line.strip().split()
                if line[0] not in miss_file_names:
                    fw.write(f"{line[0]}\t{line[1]}\n")







