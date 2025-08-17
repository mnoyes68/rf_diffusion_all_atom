import json
import os
import sys

from pathlib import Path

import numpy as np


HEADER = "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id"


def get_active_site_h_residue(trb_file):
    trb = np.load(trb_file, allow_pickle=True)
    return trb['con_hal_pdb_idx'][0][1]


def constraint_file_text(hres):
    return f"A,H{hres}@N,B,@O6B,covalent,1.0,0.0,0.0,protein-glycan,bond1"


if __name__ == '__main__':
    backbone_dir = sys.argv[1]
    output_dir = sys.argv[2]

    backbone_dir_path = Path(backbone_dir)
    matching_files = []
    
    # Get all .pdb files with a corresponding trb file
    pdb_files = backbone_dir_path.glob('*.pdb')
    for pdb_file in pdb_files:
        trb_file = pdb_file.with_suffix('.trb')
        if trb_file.exists():
            matching_files.append((pdb_file, trb_file))

    for pdb_file, trb_file in matching_files:
        # Write the restraints file
        with open(os.path.join(output_dir, f"{pdb_file.stem}.restraints"), "w") as f:
            f.write(HEADER + "\n")
            f.write(constraint_file_text(get_active_site_h_residue(trb_file)))
