import ast
import json
import os
import sys
import numpy as np


def get_fixed_site(trbfile):
    fixed_sites = {}
    trb = np.load(trbfile, allow_pickle=True)
    indices = list(np.where(trb['inpaint_str'])[0])
    active_site = " ".join([f"A{i+1}" for i in indices])
    return active_site


if __name__ == "__main__":
    fixed_sites = {}
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    for f in os.listdir(input_dir):
        pdbfile = os.path.join(input_dir, f.strip())
        if not pdbfile.endswith('.pdb'):
            continue
        trbfile = pdbfile.replace('.pdb', '.trb')
        if not os.path.exists(trbfile):
            continue
        fixed_sites[pdbfile] = get_fixed_site(trbfile)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fixed_sites, f, indent=4, ensure_ascii=False)
