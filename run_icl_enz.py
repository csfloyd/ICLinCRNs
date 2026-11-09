#################################################
################  Import things #################
#################################################


import torch
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader
import os
import time

# Import from refactored modular structure
from data_generation import GaussianMixtureModel, generate_icl_gmm_data, build_input_projection_matrices
from datasets import ICLGMMDataset, collate_fn
from models import *  # includes EnzymeCascadeICL
from training import train_model, train_models_joint
from evaluation import test_icl


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments

parser.add_argument("--param1", type=int, required=True, help="N: number of context examples")
parser.add_argument("--param2", type=int, required=False, 
                    help="rate_anchor_lambda for the 'enzyme_frozen' branch "
                         "(0 = rates hard-frozen, >0 = rates softly anchored to init)")
parser.add_argument("--param3", type=int, required=True, help="Random seed (also the wiring seed)")
parser.add_argument("--output", type=str, required=True, help="Output directory")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

# ============================================================
# Data Generation Parameters
# ============================================================
L = 128                      # Number of output classes
K = L                      # Number of GMM classes for data generation
D = 1                        # Dimension of input features
N = args.param2               # Number of context examples per task
B = 1                        # Burstiness parameter (zipfian sampling weight)
epsilon = 1e-3               # Within-class noise (standard deviation)
seed = args.param3           # Random seed for reproducibility
exact_copy = True            # If True, query is exact copy of a context item
shuffle_context = True       # Whether to shuffle context order during training
offset = 0.0                 # Offset applied to GMM centers
min_max_choice = None        # Optional constraint on min/max class indices
unique_labels = False        # If True, ensure all context labels are unique
equal_norm = False           # If True, normalize all GMM class means to equal norm

# ============================================================
# Input Projection Parameters
# ============================================================
input_proj_mode = "identity"   # "identity" or "random"
input_proj_scale = 1
input_proj_dim = D

# ============================================================
# Enzyme cascade parameters (graph forward)
# Receptors = (N+1)*D, output layer = N nodes, softmax over outputs = attention
# ============================================================
hidden_dim = args.param1
hidden_depth = 2
p_f = 1.0
input_scale = 5.0            # receptor conc = exp(input_scale * z)
exp_input = True             # False -> receptor conc = input_scale * z (no exp)
rate_init_std = 2.5          # rates init as exp(randn * std) (rate_init_mode="random" only)
l0_init = 1.0
structure_seed = seed

# Rate initialisation: "random" = log-normal rates (rate_init_std); "signed" = every edge has the
# same strength and a random sign (activating: kf=edge_strength, kr~0; inactivating: the reverse)
rate_init_mode = "signed"
edge_strength = 10.0         # signed mode: 'on' channel strength per edge
p_activating = 0.5           # signed mode: fraction of activating edges
signed_off_ratio = 1e-3      # signed mode: 'off' channel = edge_strength * ratio
signed_node_rate = 1.0       # signed mode: basal kf_node = kr_node

# Anchor penalty on log-rates, applied to the 'enzyme_frozen' branch only
# (lambda * mean (log k - log k_init)^2 over active rates). The 'enzyme' branch uses lambda = 0.
rate_anchor_lambda = 0.1 #args.param2

learn_rates = True           # 'enzyme' branch: train rates + l0 (lambda = 0, rates completely free)
learn_l0 = True
add_enzyme_frozen = False     # also train the anchored/frozen variant (lambda = rate_anchor_lambda)
add_enzyme_3state = False    # also train S0<->S1<->S2 nodes, active form = S2 (extra kf2/kr2 rates)

# ============================================================
# Baselines
# ============================================================
add_qk = False
qk_dim = input_proj_dim       # Key/query latent dimension for QK baseline

add_mlp = True
mlp_baseline_depth = 3
mlp_baseline_width = hidden_dim
mlp_baseline_dropout = 0.0
mlp_baseline_activation = "tanh"

# ============================================================
# Training Parameters
# ============================================================
epochs = 1000                # Number of training epochs
lr = 0.0025                  # Learning rate
batch_size = 50              # Batch size for training
train_samples = 25000        # Number of training samples
val_samples = 5000           # Number of validation samples

# ============================================================
# Inference Parameters
# ============================================================
method = 'direct_solve'      # ignored by the enzyme model
temperature = 1.0            # Softmax temperature for attention


# Set parameters
params = {
    # Data generation
    'K': K,
    'L': L,
    'D': D,
    'N': N,
    'B': B,
    'epsilon': epsilon,
    'seed': seed,
    'exact_copy': exact_copy,
    'shuffle_context': shuffle_context,
    'offset': offset,
    'min_max_choice': min_max_choice,
    'unique_labels': unique_labels,
    'equal_norm': equal_norm,

    # Enzyme cascade
    'hidden_dim': hidden_dim,
    'hidden_depth': hidden_depth,
    'p_f': p_f,
    'input_scale': input_scale,
    'exp_input': exp_input,
    'rate_init_std': rate_init_std,
    'l0_init': l0_init,
    'structure_seed': structure_seed,
    'rate_init_mode': rate_init_mode,
    'edge_strength': edge_strength,
    'p_activating': p_activating,
    'signed_off_ratio': signed_off_ratio,
    'signed_node_rate': signed_node_rate,
    'rate_anchor_lambda': rate_anchor_lambda,
    'learn_rates': learn_rates,
    'learn_l0': learn_l0,
    'add_enzyme_frozen': add_enzyme_frozen,
    'add_enzyme_3state': add_enzyme_3state,

    # Baselines
    'add_qk': add_qk,
    'add_mlp': add_mlp,
    'qk_dim': qk_dim,
    'mlp_baseline_depth': mlp_baseline_depth,
    'mlp_baseline_width': mlp_baseline_width,
    'mlp_baseline_dropout': mlp_baseline_dropout,
    'mlp_baseline_activation': mlp_baseline_activation,

    # Training
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'train_samples': train_samples,
    'val_samples': val_samples,

    # Inference
    'method': method,
    'temperature': temperature,

    # Input projection
    'input_proj_mode': input_proj_mode,
    'input_proj_scale': input_proj_scale,
    'input_proj_dim': input_proj_dim,
}


print("="*70)
print("ENZYME CASCADE ICL - CLASSIFICATION (graph forward, softmax output)")
print("="*70)
print(f"K={params['K']}, D={params['D']}, N={params['N']}, B={params['B']}")
print(f"Cascade: hidden {params['hidden_dim']}x{params['hidden_depth']} -> {params['N']} outputs, p_f={params['p_f']}, "
      f"input_scale={params['input_scale']}, exp_input={params['exp_input']}")
print(f"Rate init: {params['rate_init_mode']} "
      + (f"(edge_strength={params['edge_strength']}, p_activating={params['p_activating']}, "
         f"off_ratio={params['signed_off_ratio']}, node_rate={params['signed_node_rate']})"
         if params['rate_init_mode'] == 'signed' else f"(std={params['rate_init_std']})"))
print(f"Variants: enzyme(learn_rates={params['learn_rates']}, lambda=0), "
      f"enzyme_frozen={params['add_enzyme_frozen']} (lambda={params['rate_anchor_lambda']}), "
      f"enzyme_3state={params['add_enzyme_3state']}")
print(f"Baselines enabled: add_qk={params['add_qk']}, add_mlp={params['add_mlp']}")
print(f"Method: {params['method']}, Temperature: {params['temperature']}")
print("="*70)

# Set random seeds
torch.manual_seed(params['seed'])
np.random.seed(params['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Build optional context/query projection matrices
K_proj, Q_proj = build_input_projection_matrices(
    D=params['D'],
    proj_dim=params['input_proj_dim'],
    mode=params['input_proj_mode'],
    scale=params['input_proj_scale']
)
if params['input_proj_mode'] == 'random':
    print("Using random full-rank input projections for context/query.")

# Create GMM with discrete labels (1 to L)
print("Creating GMM with discrete labels...")
gmm = GaussianMixtureModel(
    K=params['K'],
    D=params['D'],
    L=params['L'],
    epsilon=params['epsilon'],
    seed=params['seed'],
    offset=params['offset'],
    equal_norm=params['equal_norm'],
)
print(f"  GMM: {params['K']} classes with labels randomly assigned from {{1, ..., {params['L']}}}")
print(f"  First 10 class labels: {gmm.class_to_label[:min(10, params['K'])].numpy()}")

# Generate data
print("\nGenerating data...")
train_data = generate_icl_gmm_data(gmm, params['train_samples'], params['N'],
                                   novel_classes=False, exact_copy=params['exact_copy'],
                                   B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'],
                                   min_max_choice=params['min_max_choice'], unique_labels=params['unique_labels'],
                                   K_proj=K_proj, Q_proj=Q_proj)
val_data = generate_icl_gmm_data(gmm, params['val_samples'], params['N'],
                                 novel_classes=False, exact_copy=params['exact_copy'],
                                 B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'],
                                 min_max_choice=params['min_max_choice'], unique_labels=params['unique_labels'],
                                 K_proj=K_proj, Q_proj=Q_proj)

train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=params['batch_size'],
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=params['batch_size'],
                       collate_fn=collate_fn)

# Create models
print("\nCreating models...")
enzyme_kwargs = dict(
    z_dim=params['input_proj_dim'], L=params['L'], N=params['N'],
    hidden_dim=params['hidden_dim'], hidden_depth=params['hidden_depth'], p_f=params['p_f'],
    input_scale=params['input_scale'], exp_input=params['exp_input'],
    rate_init_std=params['rate_init_std'], l0_init=params['l0_init'],
    rate_init_mode=params['rate_init_mode'], edge_strength=params['edge_strength'],
    p_activating=params['p_activating'], signed_off_ratio=params['signed_off_ratio'],
    signed_node_rate=params['signed_node_rate'],
    structure_seed=params['structure_seed'],
)
models_to_train = {
    # Rates completely free (anchor lambda = 0).
    'enzyme': EnzymeCascadeICL(
        learn_rates=params['learn_rates'], learn_l0=params['learn_l0'], **enzyme_kwargs),
}
if params['add_enzyme_frozen']:
    # Same wiring and same initial rates as 'enzyme'.
    #   lambda == 0 -> rates hard-frozen, only l0 trained
    #   lambda  > 0 -> rates trainable but softly anchored to their init, l0 free
    models_to_train['enzyme_frozen'] = EnzymeCascadeICL(
        learn_rates=(params['rate_anchor_lambda'] > 0), learn_l0=params['learn_l0'],
        rate_anchor_lambda=params['rate_anchor_lambda'], **enzyme_kwargs)
if params['add_enzyme_3state']:
    # Same wiring and same step-1 rates as 'enzyme'; adds a second phosphorylation step.
    models_to_train['enzyme_3state'] = EnzymeCascadeICL(
        learn_rates=params['learn_rates'], learn_l0=params['learn_l0'], n_phos_states=3, **enzyme_kwargs)

if params['add_qk']:
    models_to_train['qk'] = QKICL(
        z_dim=params['input_proj_dim'],
        L=params['L'],
        N=params['N'],
        d_k=params['qk_dim']
    )
if params['add_mlp']:
    models_to_train['mlp'] = MLPICL(
        z_dim=params['input_proj_dim'],
        L=params['L'],
        N=params['N'],
        depth=params['mlp_baseline_depth'],
        hidden_width=params['mlp_baseline_width'],
        dropout=params['mlp_baseline_dropout'],
        activation=params['mlp_baseline_activation'],
    )

# Train with ICL/IWL tracking
start_time = time.time()
print("\nTraining...")
print("="*70)
history = train_models_joint(
    models_to_train, train_loader, val_loader, device,
    n_epochs=params['epochs'], lr=params['lr'],
    method=params['method'], temperature=params['temperature'],
    gmm=gmm, N=params['N'], B=params['B'],
    L=params['L'], exact_copy=params['exact_copy'],
    eval_frequency=1, n_eval_samples=500, min_max_choice=params['min_max_choice'],
    unique_labels=params['unique_labels'], K_proj=K_proj, Q_proj=Q_proj
)

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Final drift of the kinetic rates from their initial values (log units)
rate_drift = {}
for name, model in models_to_train.items():
    if hasattr(model, 'rate_drift'):
        rate_drift[name] = model.rate_drift()
        print(f"{name:>14s} rate drift (log): {rate_drift[name]}  anchor_lambda={model.rate_anchor_lambda}")

# Test
results = {}
for name, model in models_to_train.items():
    # Re-seed before each test call so all models are evaluated on the same sampled tasks.
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    results[name] = test_icl(model, gmm, params['N'], device, n_samples=1000,
                             exact_copy=params['exact_copy'], B=params['B'],
                             method=params['method'], L=params['L'],
                             temperature=params['temperature'], shuffle_context=params['shuffle_context'],
                             min_max_choice=params['min_max_choice'], unique_labels=params['unique_labels'],
                             K_proj=K_proj, Q_proj=Q_proj)

# Save results
os.makedirs(output_dir, exist_ok=True)

# Save model weights (small, portable)
for name, model in models_to_train.items():
    if name == 'enzyme':
        model_path = f'{output_dir}/model.pt'
    else:
        model_path = f'{output_dir}/model_{name}.pt'
    torch.save(model.state_dict(), model_path)

# Save results and metadata (for analysis)
results_data = {
    'results': results,
    'history': history,          # includes per-epoch rate_drift_{mean_abs,rms,max} for enzyme models
    'rate_drift': rate_drift,
    'params': params,
    'K_proj': K_proj,
    'Q_proj': Q_proj,
    'execution_time': end_time - start_time
}
results_path = f'{output_dir}/results.pkl'
with open(results_path, "wb") as file:
    pickle.dump(results_data, file)

print("")
for name in models_to_train.keys():
    if name == 'enzyme':
        print(f"✓ Saved {name.upper()} model to {output_dir}/model.pt")
    else:
        print(f"✓ Saved {name.upper()} model to {output_dir}/model_{name}.pt")
print(f"✓ Saved results to {results_path}")
print(f"✓ Execution Time: {end_time - start_time:.2f} seconds")
