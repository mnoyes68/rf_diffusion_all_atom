import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from Bio.PDB import PDBParser, is_aa
from tqdm.auto import tqdm


def extract_backbone_data(pdb_path, ptm, chain_id='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)

    samples = []

    for model_idx, model in enumerate(structure):
        coords = []

        if chain_id not in model:
            continue

        for residue in model[chain_id]:
            if is_aa(residue, standard=True):
                try:
                    n = residue["N"].get_coord()
                    ca = residue["CA"].get_coord()
                    c = residue["C"].get_coord()
                    coords.extend([n, ca, c])
                except KeyError:
                    continue

        if coords:
            sample = {
                "coords": np.array(coords),
                "model_id": model_idx,
                "ptm": ptm
            }
            samples.append(sample)

    return samples


def process_df(df, backbone_directory, output_path="backbone_dataset.npy", chain_id="A"):
    all_samples = []

    for i, row in df.iterrows():
        for d in os.listdir(backbone_directory):
            pdb_path = os.path.join(d, row['backbone'])
            if os.path.exists(pdb_path):
                print(f"Processing: {pdb_path}")
                samples = extract_backbone_data(pdb_path, row['ptm'], chain_id)
                all_samples.extend(samples)

    np.save(all_samples, output_path)
    print(f"Saved {len(all_samples)} samples to {output_path}")


if __name__ == '__main__':
    input_csv = sys.argv[1]
    backbone_directory = sys.argv[2]
    output_path = sys.argv[3]
    df = pd.read_csv(input_csv)
    process_df(df, backbone_directory, output_path=output_path)