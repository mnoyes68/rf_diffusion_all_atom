#!/bin/bash
#SBATCH --partition=gpu            # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gres=gpu:1               # Number and type of GPUs
#SBATCH --cpus-per-task=2          # Number of CPU cores per task
#SBATCH --mem=32GB                 # Memory allocation
#SBATCH --time=08:00:00            # Maximum runtime (HH:MM:SS)


INPUT_PATH=$1
PDB_PATH_MULTI=$2
OUTPUT_DIR=$3

source ~/miniconda3/bin/activate
conda activate ligandmpnn_env
cd $INPUT_PATH

if [ ! -d $OUTPUT_DIR ]
then
    mkdir -p $OUTPUT_DIR
fi

TEMPERATURE_VALUES=(0.1 0.2)
for TEMPERATURE in "${TEMPERATURE_VALUES[@]}"; do
    echo "TEMPERATURE=$TEMPERATURE"
    FILE_NUMBER=$(printf "%.0f" "$(echo "$TEMPERATURE * 10" | bc)")
    python run.py \
        --model_type ligand_mpnn \
        --out_folder $OUTPUT_DIR \
        --number_of_batches 5 \
        --temperature $TEMPERATURE \
        --pdb_path_multi $PDB_PATH_MULTI \
        --fixed_residues_multi $PDB_PATH_MULTI \
        --seed 68 \
        --file_ending "_T$FILE_NUMBER" \
        --checkpoint_ligand_mpnn /projects/gllab/noyes.m/LigandMPNN/model_params/ligandmpnn_v_32_020_25.pt \
        --ligand_mpnn_use_side_chain_context 1
done
