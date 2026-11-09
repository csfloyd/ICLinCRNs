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
from models import *
from training import train_model, train_models_joint
from evaluation import test_icl


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments

parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=float, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

# ============================================================
# Data Generation Parameters
# ============================================================
L = 128                      # Number of output classes
K = L                      # Number of GMM classes for data generation
D = 4                        # Dimension of input features
N = args.param1               # Number of context examples per task
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
# Model Architecture Parameters
# ============================================================
n_nodes = 5                  # Number of nodes in the Markov chain
transform_func = 'exp'       # 'exp', 'relu', 'elu', 'softplus', 'sigmoid', or 'identity'
learn_base_rates = True      # If True, allow gradient updates to unmasked base rates

rate_encoder_type = "linear"  # 'linear' (K·z) or 'mlp' (z → n_nodes² log-rate shift)
encoder_mlp_depth = 2
encoder_mlp_width = 32
rate_decoder_type = "linear"  # 'linear' (π @ B) or 'mlp' decoder head
decoder_mlp_depth = 2
decoder_mlp_width = 32

# Detailed-balance split (linear encoder only). "directed" keeps the original model.
rate_param = "db_split"      # "directed" or "db_split"
db_phi0 = args.param2                # Hinge threshold on each force scalar
db_lambda = 100.0            # Weight on DB hinge penalty (ignored unless db_split)

# Node map M(z) -> n-vector, and which of K / B train.
# hidden_map: 'crn' (Wπ=0), 'linear' (column sums of transform(M) if LINEAR_HIDDEN_MAP_APPLY_TRANSFORM else M),
#             'relu' (ReLU of column sums of M),
#             'frozen_mlp' (frozen MLP on off-diagonal transform(M) if FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM else M)
# Freeze encoder: random CRN features, train readout. Freeze decoder: train CRN, random B.
hidden_map = "crn"
hidden_mlp_depth = 2
hidden_mlp_width = 8
learn_rate_encoder = True
learn_rate_decoder = True

add_qk = False
qk_dim = input_proj_dim       # Key/query latent dimension for QK baseline

add_mlp = False
mlp_baseline_depth = 2
mlp_baseline_width = 32
mlp_baseline_dropout = 0.0
mlp_baseline_activation = "relu"

# ============================================================
# Sparsity Parameters - K_params (context-dependent modulation)
# ============================================================
sparsity_rho_edge = 1.0       # Fraction of (i,j) edges with K parameters
sparsity_rho_all = 1.0       # Fraction of individual K parameters to keep

# ============================================================
# Sparsity Parameters - Base Rates
# ============================================================
sparsity_rho_edge_base_W = 1.0   # Fraction of (i,j) edges with base rates in W
base_mask_value = float('-inf')            # Value for masked base rates: 0.0 (no bias) or float('-inf') (disable edge)

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
method = 'direct_solve'      # Steady-state solver: 'direct_solve', 'matrix_tree', or 'linear_solver'
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
    
    # Model architecture
    'n_nodes': n_nodes,
    'add_qk': add_qk,
    'add_mlp': add_mlp,
    'transform_func': transform_func,
    'learn_base_rates': learn_base_rates,
    'rate_encoder_type': rate_encoder_type,
    'encoder_mlp_depth': encoder_mlp_depth,
    'encoder_mlp_width': encoder_mlp_width,
    'rate_decoder_type': rate_decoder_type,
    'decoder_mlp_depth': decoder_mlp_depth,
    'decoder_mlp_width': decoder_mlp_width,
    'rate_param': rate_param,
    'db_phi0': db_phi0,
    'db_lambda': db_lambda,
    'hidden_map': hidden_map,
    'hidden_mlp_depth': hidden_mlp_depth,
    'hidden_mlp_width': hidden_mlp_width,
    'learn_rate_encoder': learn_rate_encoder,
    'learn_rate_decoder': learn_rate_decoder,
    'qk_dim': qk_dim,
    'mlp_baseline_depth': mlp_baseline_depth,
    'mlp_baseline_width': mlp_baseline_width,
    'mlp_baseline_dropout': mlp_baseline_dropout,
    'mlp_baseline_activation': mlp_baseline_activation,
    
    # Sparsity - K_params
    'sparsity_rho_edge': sparsity_rho_edge,
    'sparsity_rho_all': sparsity_rho_all,
    
    # Sparsity - Base rates
    'sparsity_rho_edge_base_W': sparsity_rho_edge_base_W,
    'base_mask_value': base_mask_value,
    
    # Training
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'train_samples': train_samples,
    'val_samples': val_samples,
    
    # Inference
    'method': method,
    'temperature': temperature
    ,
    # Input projection
    'input_proj_mode': input_proj_mode,
    'input_proj_scale': input_proj_scale,
    'input_proj_dim': input_proj_dim
}


print("="*70)
print("MARKOV ICL - CLASSIFICATION (Softmax Output)")
print("="*70)
print(f"K={params['K']}, D={params['D']}, N={params['N']}, B={params['B']}, nodes={params['n_nodes']}")
print(f"Data options: equal_norm={params['equal_norm']}, epsilon={params['epsilon']}")
print(f"Baselines enabled: add_qk={params['add_qk']}, add_mlp={params['add_mlp']}")
print(f"Method: {params['method']}, Temperature: {params['temperature']}")
print(
    f"Rate encoder: {params['rate_encoder_type']} "
    f"(mlp depth/width={params['encoder_mlp_depth']}/{params['encoder_mlp_width']} if mlp)"
)
print(
    f"Rate decoder: {params['rate_decoder_type']} "
    f"(mlp depth/width={params['decoder_mlp_depth']}/{params['decoder_mlp_width']} if mlp)"
)
print(f"Rate param: {params['rate_param']}")
if params['rate_param'] == 'db_split':
    print(f"DB constraint: phi0={params['db_phi0']}, lambda={params['db_lambda']}")
print(
    f"Hidden map: {params['hidden_map']}, "
    f"learn encoder/decoder={params['learn_rate_encoder']}/{params['learn_rate_decoder']}"
)
if params['hidden_map'] == 'frozen_mlp':
    print(
        f"Frozen hidden MLP: depth={params['hidden_mlp_depth']}, "
        f"width={params['hidden_mlp_width']} "
        f"(on off-diagonal {params['transform_func']}(M) if FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM else M)"
    )
if params['hidden_map'] == 'linear':
    print(f"Linear hidden map transform: {params['transform_func']}")
print(
    f"Input projection: mode={params['input_proj_mode']}, "
    f"dim={params['input_proj_dim']}, scale={params['input_proj_scale']}"
)
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
                                   min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'],
                                   K_proj=K_proj, Q_proj=Q_proj)
val_data = generate_icl_gmm_data(gmm, params['val_samples'], params['N'], 
                                 novel_classes=False, exact_copy=params['exact_copy'], 
                                 B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'],
                                 min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'],
                                 K_proj=K_proj, Q_proj=Q_proj)

train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=params['batch_size'],
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=params['batch_size'],
                       collate_fn=collate_fn)

if params['rate_param'] == 'db_split' and params['rate_encoder_type'] != 'linear':
    print("rate_param='db_split' requires a linear rate encoder; overriding rate_encoder_type='linear'.")
    params['rate_encoder_type'] = 'linear'

# Create models
print("\nCreating model...")
markov_model = MatrixTreeMarkovICL(
    n_nodes=params['n_nodes'],
    z_dim=params['input_proj_dim'],
    L=params['L'],
    N=params['N'],
    learn_base_rates=params['learn_base_rates'],
    transform_func=params['transform_func'],
    sparsity_rho_edge=params['sparsity_rho_edge'],
    sparsity_rho_all=params['sparsity_rho_all'],
    sparsity_rho_edge_base_W=params['sparsity_rho_edge_base_W'],
    base_mask_value=params['base_mask_value'],
    rate_encoder_type=params['rate_encoder_type'],
    encoder_mlp_depth=params['encoder_mlp_depth'],
    encoder_mlp_width=params['encoder_mlp_width'],
    rate_decoder_type=params['rate_decoder_type'],
    decoder_mlp_depth=params['decoder_mlp_depth'],
    decoder_mlp_width=params['decoder_mlp_width'],
    rate_param=params['rate_param'],
    db_phi0=params['db_phi0'],
    db_lambda=params['db_lambda'],
    hidden_map=params['hidden_map'],
    hidden_mlp_depth=params['hidden_mlp_depth'],
    hidden_mlp_width=params['hidden_mlp_width'],
    learn_rate_encoder=params['learn_rate_encoder'],
    learn_rate_decoder=params['learn_rate_decoder'],
)
models_to_train = {'markov': markov_model}

if params['add_qk']:
    qk_model = QKICL(
        z_dim=params['input_proj_dim'],
        L=params['L'],
        N=params['N'],
        d_k=params['qk_dim']
    )
    models_to_train['qk'] = qk_model
if params['add_mlp']:
    mlp_model = MLPICL(
        z_dim=params['input_proj_dim'],
        L=params['L'],
        N=params['N'],
        depth=params['mlp_baseline_depth'],
        hidden_width=params['mlp_baseline_width'],
        dropout=params['mlp_baseline_dropout'],
        activation=params['mlp_baseline_activation'],
    )
    models_to_train['mlp'] = mlp_model

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
    if name == 'markov':
        model_path = f'{output_dir}/model.pt'
    else:
        model_path = f'{output_dir}/model_{name}.pt'
    torch.save(model.state_dict(), model_path)

# Save results and metadata (for analysis)
results_data = {
    'results': results,
    'history': history,
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
    if name == 'markov':
        print(f"✓ Saved {name.upper()} model to {output_dir}/model.pt")
    else:
        print(f"✓ Saved {name.upper()} model to {output_dir}/model_{name}.pt")
print(f"✓ Saved results to {results_path}")
print(f"✓ Execution Time: {end_time - start_time:.2f} seconds")