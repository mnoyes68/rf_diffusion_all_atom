# RFDiffusionAA with Resampling

This repository is a fork of [RFDiffusion All-Atom](https://github.com/baker-laboratory/rf_diffusion_all_atom), a protein diffusion model used for generating _de novo_ protein backbones. This repository has been forked to implement resampling using particle filtering. This is the code repository containing the code from the paper **Resampling Techniques to Improve RFDiffusion Results for de novo Design of an Enzyme Based on Chondroitinase ABC Lyase I**.

## Installation

Follow the script below to install this environment. Note that it is recommended to use `python3.9` for this installation, though it may work with other versions depending on your CUDA version.

```
python3.9 -m venv rfdenv
source rfdenv/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install omegaconf hydra-core scipy icecream openbabel-wheel assertpy opt_einsum pandas pydantic deepdiff e3nn fire scikit-learn torchdata==0.9.0
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu121/repo.html
```

Alternatively, if you are using CUDA 12.1, you can run `pip install -r requirements.txt` to get started.

### Additional Models

The pipeline in the paper additionally specifies LigandMPNN and Chai1 as models to use for the downstream steps. They are not included in this repository, but if you wish to follow the full pipeline, follow the instructions for installing the models below.
- [LigandMPNN](https://github.com/dauparas/LigandMPNN)
- [Chai1](https://github.com/chaidiscovery/chai-lab)

Note that LigandMPNN expects to be installed via `conda`. Chai can be installed with `pip` but requires Python >= 3.10. If using the pipeline below, the sbatch files will expect to be able to find the following environments for these models
- `ligandmpnn_env`
- `chai`

## Pipeline

In order to run the pipeline from the paper, follow the steps below. Note that this is assumed to be ran on a shared cluster using SLURM, that can take in `sbatch` files. If this is being ran on a different environment, the commands in the `.sbatch` files should still be valid with some adjustment.

```
# Call with argument syntax:
# - 1: Path to repository
# - 2: Output prefix
# - 3: Design start number
# - 4: Number of particles
sbatch examples/backbone_generation.sbatch <path/to/rf_diffusion_all_atom> output/test 0 16

# Prepare the input for LigandMPNN
python get_fixed_sites.py ../LigandMPNN output/ fixed_sites.json

# Run LigandMPNN
sbatch examples/sequence_recovery.sbatch <path/to/ligand_mpnn> fixed_sites.json sequences/

# Prepare the input for Chai1
python scripts/prepare_chai_input.py sequences/ chai_input.csv fa_files/

# If running with a ligand, create a constraints file to represent the connection between structure/ligand.
# Currently this is hardcoded to represent the ligand in 7yke.
python scripts/create_constraints.py output/ fa_files/

# Run Chai1
sbatch examples/structure_prediction.sbatch <path/to/rf_diffusion_all_atom> chai_input.csv fa_files/ predictions/ 0

# Get the Chai1 Output
python scripts/parse_chai_output.py predictions/ chai_output.csv
```

## Acknowledgements

Thank you to @w-ahern, @r-krishna and @ikalvet for your hard work on RFDiffusion All-Atom, which made this research possible.
