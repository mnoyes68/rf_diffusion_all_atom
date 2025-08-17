import csv
import re
import os
import sys
import numpy as np

SCORE_FILE_PATTERN = r"scores\.model_idx_\d+\.npz"


def get_top_scores(score_dir):
    top_ptm = 0
    top_iptm = 0
    score_files = [f for f in os.listdir(score_dir) if re.match(SCORE_FILE_PATTERN, f)]
    for f in score_files:
        data = np.load(os.path.join(score_dir, f))
        ptm = data['ptm'][0]
        iptm = data['iptm'][0]
        top_ptm = max(top_ptm, ptm)
        top_iptm = max(top_iptm, iptm)

    return top_ptm, top_iptm



if __name__ == '__main__':
    score_dir = sys.argv[1]
    output_csv = sys.argv[2]

    data = []
    for d in os.listdir(score_dir):
        ptm, iptm = get_top_scores(os.path.join(score_dir, d))
        data.append({"name": d, "ptm": ptm, "iptm": iptm})

    with open(output_csv, mode="w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "ptm", "iptm"])
        
        writer.writeheader()  # Write column headers
        writer.writerows(data)
        
