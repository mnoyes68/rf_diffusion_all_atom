#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00

INPUT_PATH=$1
EVALCSV=$2
INDIR=$3
OUTDIR=$4
OFFSET=$5

cd $INPUT_PATH
source chai/bin/activate

XYZFILES=$(scripts/python print_offset_files.py $EVALCSV 1000 $OFFSET True)
for file in $XYZFILES; do
  prefix="${file%%_T*}"
  output=$OUTDIR/${file%.fa}
  mkdir $output
  chai-lab fold $INDIR/$file --constraint-path $INDIR/$prefix.restraints $output
  # Unconstrained mode
  # chai-lab fold $INDIR/$file $output
done


