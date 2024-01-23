#!/usr/bin/env python3

import sys

def tsv2list(file: str):
    with open(file, 'r')as f1:
        folder=''
        path=''
        list1=[]
        for i, line in enumerate(f1):
            line = line.strip()
            if i == 0:
                line = line.split("/")
                path = '/'.join(line[:-1])
                folder = line[-1]
                list1.append(path)
            else:
                content = folder + '/' + line
                list1.append(content)
        return list1



if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    #file3 = sys.argv[3]
    output = sys.argv[3]
    list1 = tsv2list(file1)
    list2 = tsv2list(file2)
    #list3 = tsv2list(file3)
 

    with open(output, 'w') as fw:
        print(list1[0],file=fw)
        for line in list1[1:]:
            print(line,file=fw)
        for line in list2[1:]:
            print(line,file=fw)
        #for line in list3[1:]:
        #    print(line,file=fw)

