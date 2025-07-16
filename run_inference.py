"""
Inference script.

To run with aa.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name other_config

where other_config can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import os

import re
import os, time, pickle
import dataclasses
import torch 
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from util import writepdb_multi, writepdb
from inference import utils as iu
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
import inference.model_runners
import rf2aa.tensor_util
import idealize_backbone
import rf2aa.util
import aa_model
import copy

import e3nn.o3 as o3

def warm_up_spherical_harmonics():
    ''' o3.spherical_harmonics returns different values on 1st call vs all subsequent calls
    All subsequent calls are reproducible.
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    relative_pos = torch.tensor([[1.,1.,1.], [1.,1.,1.]]).to(device).to(torch.float32)
    sh1 = o3.spherical_harmonics([1,2,3], relative_pos, normalize=True)
    sh2 = o3.spherical_harmonics([1,2,3], relative_pos, normalize=True)

def make_deterministic(seed=0):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    seed_all(seed)
    warm_up_spherical_harmonics()

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seeds():
    return {
        'torch': torch.get_rng_state(),
        'np': np.random.get_state(),
        'python': random.getstate(),
    }

@hydra.main(version_base=None, config_path='config/inference', config_name='aa')
def main(conf: HydraConfig) -> None:
    sampler = get_sampler(conf)
    sample(sampler, conf)

def get_sampler(conf):
    if conf.inference.deterministic:
        make_deterministic()

    # Loop over number of designs to sample.
    design_startnum = conf.inference.design_startnum
    if conf.inference.design_startnum == -1:
        existing = glob.glob(conf.inference.output_prefix + '*.trb')
        indices = [-1]
        for e in existing:
            m = re.match(r'.*_(\d+)\.trb$', e)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1   

    conf.inference.design_startnum = design_startnum
    # Initialize sampler and target/contig.
    sampler = inference.model_runners.sampler_selector(conf)
    return sampler


class Particle:
    """
    A particle is a unit to sampled via the diffusion process. At each iteration
    the `resample` method will duplicate high weight particles and drop low weight
    particles. The `clone` method must be kept up to date with all of the relevant
    componets for sampling.
    """

    def __init__(
        self,
        indep,
        is_diffused,
        is_seq_masked,
        denoiser,
        contig_map,
        denoised_xyz_stack=None,
        px0_xyz_stack=None,
        seq_stack=None,
        rfo=None,
        prev_twist=None,
        new_twist=None,
        opt_lp=None,
        cond_lp=None,
        log_weight=None
    ):
        self.indep = indep
        self.is_diffused = is_diffused
        self.is_seq_masked = is_seq_masked
        self.denoiser = denoiser
        self.contig_map = contig_map

        # These will be populated during sampling
        self.denoised_xyz_stack = denoised_xyz_stack if denoised_xyz_stack else []
        self.px0_xyz_stack = px0_xyz_stack if px0_xyz_stack else []
        self.seq_stack = seq_stack if seq_stack else []
        self.rfo = rfo

        # Stats from sampling
        self.prev_twist = prev_twist
        self.new_twist = new_twist
        self.opt_lp = opt_lp
        self.cond_lp = cond_lp
        self.log_weight = log_weight
    
    def shift_twist(self):
        """The new twist is now the previous"""
        self.prev_twist = self.new_twist.clone()
        self.new_twist = None

    def clone(self):
        """Clone this particle to continue diffusing"""
        indep = self.indep.clone()
        is_diffused = self.is_diffused.clone()
        is_seq_masked = self.is_seq_masked.clone()
        denoiser = self.denoiser  # TBD if this will be safe or not
        contig_map = self.contig_map  # TBD if this will be safe or not

        denoised_xyz_stack = [d.clone() for d in self.denoised_xyz_stack]
        px0_xyz_stack = [p.clone() for p in self.px0_xyz_stack]
        seq_stack = [s.clone() for s in self.seq_stack]
        rfo = self.rfo.clone()

        prev_twist = self.prev_twist.clone()
        new_twist = self.new_twist.clone()
        opt_lp = self.opt_lp.clone()
        cond_lp = self.cond_lp.clone()
        log_weight = self.log_weight.clone()

        return type(self)(
            indep,
            is_diffused,
            is_seq_masked,
            denoiser,
            contig_map,
            denoised_xyz_stack,
            px0_xyz_stack,
            seq_stack,
            rfo,
            prev_twist,
            new_twist,
            opt_lp,
            cond_lp,
            log_weight
        )


def resample(particles: list[Particle]) -> list[Particle]:
    # Initial parameters for systematic resampling
    n = len(particles)
    n_inv = 1 / n
    m = float(torch.rand(1)) * n_inv
    queries = torch.linspace(m, m + ((n - 1) * n_inv), n)
    queries = torch.clamp(queries, 0.0, 1.0 - 1e-6)

    # Get the weights
    log_w = torch.Tensor([p.log_weight for p in particles])
    if torch.isnan(log_w).any():
        print("NAN Detected, skipping resampling")
        return particles
    w = torch.exp(log_w - log_w.max())
    w_sort = w.sort(descending=True)
    w = w_sort.values
    norm_w = w / w.sum()

    # Resort particles
    particles = [particles[i] for i in w_sort.indices]

    # Get the indices to resample
    cumulative = torch.cumsum(norm_w, dim=0)
    indices = torch.searchsorted(cumulative, queries, right=True)

    # Resample
    resampled_particles = []
    seen_indices = set()
    for i in indices:
        i = int(i)
        p = particles[i]

        # Clone the particle if this has already been seen
        if i in seen_indices:
            p = p.clone()

        # Save and continue
        resampled_particles.append(p)
        seen_indices.add(i)
    
    return resampled_particles


def sample(sampler, conf):
    inf = conf.inference

    log = logging.getLogger(__name__)
    des_i_start = inf.design_startnum
    des_i_end = inf.design_startnum + inf.num_designs
    for i_des in range(des_i_start, des_i_end):
        if inf.deterministic:
            seed_all(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        sampler.output_prefix = out_prefix
        log.info(f'Making batch design {i_des} of {des_i_start}:{des_i_end}')
        particles, weight_history = sample_batch(sampler, conf, i_des)
        log.info(f'Finished batch design in {(time.time()-start_time)/60:.2f} minutes')
        save_outputs(sampler, particles, weight_history)


def sample_batch(sampler, conf, i_des, simple_logging=False):
    particles = []
    for i in range(conf.inference.particles):
        indep, is_diffused, is_seq_masked, denoiser, contig_map = sampler.sample_init()
        particles.append(
            Particle(indep, is_diffused, is_seq_masked, denoiser, contig_map)
        )

    # Loop over number of reverse diffusion time steps.
    weight_history = {}
    for t in range(int(conf.diffuser.T), conf.inference.final_step-1, -1):
        if t < int(conf.diffuser.T) - 1:
            particles = resample(particles)

        for particle in particles:
            particle.shift_twist()
            indep = particle.indep
            rfo = particle.rfo
            px0, x_t, seq_t, opt_lp, cond_lp, p_t, tors_t, plddt, rfo = (
                sampler.sample_step(t, particle)
            )
            rf2aa.tensor_util.assert_same_shape(indep.xyz, x_t)
            indep.xyz = x_t
            particle.new_twist = p_t

            aa_model.assert_has_coords(indep.xyz, indep)

            particle.indep = indep
            particle.px0_xyz_stack.append(px0)
            particle.denoised_xyz_stack.append(x_t)
            particle.seq_stack.append(seq_t)
            particle.rfo = rfo
            particle.opt_lp = opt_lp
            particle.cond_lp = cond_lp
        
        # Calculate weights
        weight_stats = []

        if t >= int(conf.diffuser.T):
            # Initialize weight and move on
            for i, particle in enumerate(particles):
                particle.log_weight = particle.new_twist
        else:
            for i, particle in enumerate(particles):
                twist_prev = particle.prev_twist
                twist_new = particle.new_twist
                opt_lp = particle.opt_lp
                cond_lp = particle.cond_lp
                log_weight = opt_lp + twist_new - cond_lp - twist_prev
                particle.log_weight = log_weight
                ws = {
                    "lp_optimal": float(opt_lp),
                    "lp_conditional": float(cond_lp),
                    "twist_prev": float(twist_prev),
                    "twist_new": float(twist_new),
                    "log_weight": float(log_weight),
                }
                weight_stats.append(ws)

        if len(weight_stats) > 0:
            weight_history[t] = weight_stats

    # Flip order for better visualization in pymol
    for particle in particles:
        particle.denoised_xyz_stack = torch.stack(particle.denoised_xyz_stack)
        particle.denoised_xyz_stack = torch.flip(particle.denoised_xyz_stack, [0,])
        particle.px0_xyz_stack = torch.stack(particle.px0_xyz_stack)
        particle.px0_xyz_stack = torch.flip(particle.px0_xyz_stack, [0,])

    return particles, weight_history


def save_outputs(sampler, particles, weight_history):
    # out_prefix
    log = logging.getLogger(__name__)

    for i, particle in enumerate(particles):
        indep = particle.indep
        out_prefix = f"{sampler.output_prefix}{chr(ord('a') + i)}"
        denoised_xyz_stack = particle.denoised_xyz_stack
        px0_xyz_stack = particle.px0_xyz_stack
        seq_stack = particle.seq_stack

        final_seq = seq_stack[-1]

        if sampler._conf.seq_diffuser.seqdiff is not None:
            # When doing sequence diffusion the model does not make predictions beyond category 19
            final_seq = final_seq[:,:20] # [L,20]

        # All samplers now use a one-hot seq so they all need this step
        final_seq[~indep.is_sm, 22:] = 0
        final_seq = torch.argmax(final_seq, dim=-1)

        # replace mask and unknown tokens in the final seq with alanine
        final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)
        seq_design = final_seq.clone()
        xyz_design = denoised_xyz_stack[0].clone()

        # Determine lengths of protein and ligand for correct chain labeling in output pdb
        chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain)

        # Save outputs
        out_head, out_tail = os.path.split(out_prefix)
        unidealized_dir = os.path.join(out_head, 'unidealized')
        os.makedirs(out_head, exist_ok=True)
        os.makedirs(unidealized_dir, exist_ok=True)

        # pX0 last step
        out_unidealized = os.path.join(unidealized_dir, f'{out_tail}.pdb')
        xyz_design[particle.is_diffused, 3:] = np.nan
        aa_model.write_traj(out_unidealized, xyz_design[None,...], seq_design, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
        out_idealized = f'{out_prefix}.pdb'

        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, out_unidealized)

        # Idealize the backbone (i.e. write the oxygen at the position inferred from N,C,Ca)
        idealize_backbone.rewrite(out_unidealized, out_idealized)
        des_path = os.path.abspath(out_idealized)

        # trajectory pdbs
        traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
        os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

        out = f'{traj_prefix}_Xt-1_traj.pdb'
        aa_model.write_traj(out, denoised_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
        xt_traj_path = os.path.abspath(out)

        out=f'{traj_prefix}_pX0_traj.pdb'
        aa_model.write_traj(out, px0_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
        x0_traj_path = os.path.abspath(out)

        # run metadata
        sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
        trb = dict(
            config = OmegaConf.to_container(sampler._conf, resolve=True),
            device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
            px0_xyz_stack = px0_xyz_stack.detach().cpu().numpy(),
            indep={k:v.detach().cpu().numpy() for k,v in dataclasses.asdict(indep).items()},
            weight_history=weight_history,
        )
        if hasattr(particle, 'contig_map'):
            for key, value in particle.contig_map.get_mappings().items():
                trb[key] = value

        for out_path in des_path, xt_traj_path, x0_traj_path:
            aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, out_path)

        with open(f'{out_prefix}.trb','wb') as f_out:
            pickle.dump(trb, f_out)

        log.info(f'design : {des_path}')
        log.info(f'Xt traj: {xt_traj_path}')
        log.info(f'X0 traj: {x0_traj_path}')


if __name__ == '__main__':
    main()
