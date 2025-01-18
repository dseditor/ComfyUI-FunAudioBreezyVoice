import os
import sys

def align_trans_scp_file(trans, scp):
    trans_dict = {}
    with open(trans, 'r') as f:
        for line in f:
            sec = line.strip().split("\t")
            trans_dict[sec[0]] = sec[1]
    scp_dict = {}
    with open(scp, 'r') as f:
        for line in f:
            sec = line.strip().split(" ")
            scp_dict[sec[0]] = sec[1]
    with open("text", "w") as f:
        for k, v in scp_dict.items():
            f.write("%s\t%s\n"%(k,trans_dict[k]))

if __name__ == '__main__':
    trans = sys.argv[1]
    scp = sys.argv[2]
    align_trans_scp_file(trans, scp)