#!/bin/bash
#SBATCH --partition=gpu                   # Partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:h200:1                 # Number and type of GPUs
#SBATCH --cpus-per-task=2                 # Number of CPU cores per task
#SBATCH --mem=16GB                        # Memory allocation
#SBATCH --time=08:00:00                   # Maximum runtime (HH:MM:SS)

INPUT_PATH=$1
OUTPUT_PREFIX=$2
NUM_DESIGNS=$3
PARTICLES=$4
TRAJECTORY_MODEL_PATH=$5

cd $INPUT_PATH
source rfdaa/bin/activate
python run_inference.py \
  inference.deterministic=False \
  inference.output_prefix=$OUTPUT_PREFIX \
  inference.input_pdb=input/7yke.pdb \
  contigmap.contigs=[\'10-140,A501-501,5-15,A508-508,10-50,A560-560,10-50,A653-653,10-140\'] \
  contigmap.length="150-300" \
  inference.ligand=UNL \
  inference.num_designs=$NUM_DESIGNS \
  inference.design_startnum=0 \
  potentials.guiding_potentials=[\'type:ligand_ncontacts,weight:1\'] \
  potentials.guide_scale=2 \
  potentials.guide_decay=cubic \
  inference.particles=$PARTICLES \
  inference.trajectory_model_path=$TRAJECTORY_MODEL_PATH