import csv
import re
import os
import subprocess
import sys

from Bio import SeqIO


def write_fa_files(data, fa_dir):
    constrained_fa_dir = os.path.join(fa_dir, "constrained")
    unconstrained_fa_dir = os.path.join(fa_dir, "unconstrained")
    for p in [fa_dir, constrained_fa_dir, unconstrained_fa_dir]:
        if not os.path.exists(p):
            os.mkdir(p)
    
    for d in data:
        filename = f"{d['file']}_T{int(round(d['temperature'] * 10))}_{d['number']}"
        header = f">protein|{filename}"
        sequence = d["sequence"]
        glycan_header = ">glycan|cs4-ts"
        glycan_formula = "GCU(4-1 ASG(3-1 GCU(4-1 ASG)))"

        constrained_fa_path = os.path.join(constrained_fa_dir, filename + ".fa")
        unconstrained_fa_path = os.path.join(unconstrained_fa_dir, filename + ".fa")
        with open(constrained_fa_path, "w") as f:
            f.write(header + "\n")
            f.write(sequence + "\n")
            f.write(glycan_header + "\n")
            f.write(glycan_formula)
    
        with open(unconstrained_fa_path, "w") as f:
            f.write(header + "\n")
            f.write(sequence)   


def process_fa_files(directory):
    csv_data = []
    
    for filename in os.listdir(directory):
        if not filename.endswith(".fa"):
            continue
        file_path = os.path.join(directory, filename)
        for record in SeqIO.parse(file_path, "fasta"):
            description = record.description
            if "model_path" in description:
                continue
            id_match = re.search(r"id=(\d+)", description)
            t_match = re.search(r"T=([\d.]+)", description)
            overall_conf_match = re.search(r"overall_confidence=([\d.]+)", description)
            ligand_conf_match = re.search(r"ligand_confidence=([\d.]+)", description)
            csv_data.append({
                "file": record.id.replace(",", ""),
                "number": int(id_match.group(1)),
                "temperature": float(t_match.group(1)),
                "overall_confidence": float(overall_conf_match.group(1)),
                "ligand_confidence": float(ligand_conf_match.group(1)),
                "sequence": str(record.seq)
            })

    return csv_data



if __name__ == '__main__':
    seq_dir = sys.argv[1]
    seq_csv_path = sys.argv[2]
    fa_dir = sys.argv[3]

    # Get output CSV file
    data = process_fa_files(seq_dir)
    with open(seq_csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "file", "number", "temperature", "overall_confidence", "ligand_confidence", "sequence"
        ])
        writer.writeheader()
        writer.writerows(data)

    # Write fasta files
    write_fa_files(data, fa_dir)
