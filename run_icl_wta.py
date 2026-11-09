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
from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from datasets import ICLGMMDataset, collate_fn
from models import WinnerTakesAllICL, QKICL
from training import train_models_joint
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
K = L                        # Number of GMM classes for data generation
D = args.param1                        # Low-D so long decoys can align with the query
N = 4                        # Number of context examples per task
B = 1                        # Burstiness parameter (zipfian sampling weight)
epsilon = 1e-3               # Within-class noise (standard deviation)
seed = args.param3           # Random seed for reproducibility
exact_copy = False           # Query is another draw from the same class (nearest-mean ICL)
shuffle_context = True       # Whether to shuffle context order during training
offset = 0.0                 # Offset applied to GMM centers
use_offset = False            # If True, scale class means (and noise) by offset / sqrt(D)
mean_anisotropy = None       # Unused: this run varies mean-norm spread, not axis stretch
mean_norm_spread = 0.0 if args.param2 is None else args.param2  # extra lognormal on Gaussian means: μ *= exp(σ g)
min_max_choice = None        # Optional constraint on min/max class indices
unique_labels = False         # Distinct context labels so attention is not confounded

# ============================================================
# Model Architecture Parameters
# ============================================================
n_nodes = 10                  # Number of nodes in the WTA network
R0 = 5.0                     # Total number of resource, should be >= 1
activation = 'softplus'      # How to impose Y_j >= 0: 'softplus', 'relu', or 'soft'
f_activation = 'softplus'    # How to impose f_j >= 0
beta_softplus = 10.0         # Parameter for softplus, larger value is closer to ReLU
learn_K = True               # Whether to learn K_j parameters
learn_beta = True            # Whether to learn beta_j parameters

add_qk = True               # Train a QK softmax baseline on the same batches
qk_dim = D                   # Key/query latent dimension for QK baseline

# ============================================================
# Annealing Parameters
# ============================================================
use_annealing = False        # Whether to use temperature annealing
tau_start = 0.2              # Starting temperature for annealing
tau_end = 0.01               # Ending temperature for annealing
beta_softplus_start = 2.0    # Starting beta for softplus annealing
beta_softplus_end = 10.0     # Ending beta for softplus annealing
anneal_epochs = 100          # Number of epochs for annealing

# ============================================================
# Sparsity Parameters
# ============================================================
sparsity_rho_edge = 1.0  # Fraction of (i,j) edges with parameters
sparsity_rho_all = 1.0       # Fraction of individual parameters to keep

# ============================================================
# Training Parameters
# ============================================================
epochs = 1000                 # Number of training epochs
lr = 0.0025                  # Learning rate
batch_size = 50              # Batch size for training
train_samples = 25000        # Number of training samples
val_samples = 5000           # Number of validation samples

# ============================================================
# Inference Parameters
# ============================================================
method = 'soft'              # Inference method for WTA
temperature = 0.1            # Softmax temperature for attention


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
    'use_offset': use_offset,
    'mean_anisotropy': mean_anisotropy,
    'mean_norm_spread': mean_norm_spread,
    'min_max_choice': min_max_choice,
    'unique_labels': unique_labels,
    
    # Model architecture
    'n_nodes': n_nodes,
    'R0': R0,
    'activation': activation,
    'f_activation': f_activation,
    'beta_softplus': beta_softplus,
    'learn_K': learn_K,
    'learn_beta': learn_beta,
    'add_qk': add_qk,
    'qk_dim': qk_dim,
    
    # Annealing
    'use_annealing': use_annealing,
    'tau_start': tau_start,
    'tau_end': tau_end,
    'beta_softplus_start': beta_softplus_start,
    'beta_softplus_end': beta_softplus_end,
    'anneal_epochs': anneal_epochs,
    
    # Sparsity
    'sparsity_rho_edge': sparsity_rho_edge,
    'sparsity_rho_all': sparsity_rho_all,
    
    # Training
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'train_samples': train_samples,
    'val_samples': val_samples,
    
    # Inference
    'method': method,
    'temperature': temperature
}


print("="*70)
print("WTA ICL - CLASSIFICATION (Softmax Output)")
print("="*70)
print(f"K={params['K']}, D={params['D']}, N={params['N']}, B={params['B']}, nodes={params['n_nodes']}")
print(f"Data options: exact_copy={params['exact_copy']}, unique_labels={params['unique_labels']}, "
      f"mean_norm_spread={params['mean_norm_spread']}, epsilon={params['epsilon']}")
print(f"Baselines enabled: add_qk={params['add_qk']}")
print(f"Method: {params['method']}, Temperature: {params['temperature']}")
print("="*70)

# Set random seeds
torch.manual_seed(params['seed'])
np.random.seed(params['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create GMM with discrete labels (1 to L)
print("Creating GMM with discrete labels...")
gmm = GaussianMixtureModel(K=params['K'], D=params['D'], L=params['L'], epsilon=params['epsilon'], seed=params['seed'], offset=params['offset'], use_offset=params['use_offset'], mean_anisotropy=params['mean_anisotropy'], mean_norm_spread=params['mean_norm_spread'])
print(f"  GMM: {params['K']} classes with labels randomly assigned from {{1, ..., {params['L']}}}")
print(f"  First 10 class labels: {gmm.class_to_label[:min(10, params['K'])].numpy()}")

# Generate data
print("\nGenerating data...")
train_data = generate_icl_gmm_data(gmm, params['train_samples'], params['N'], 
                                   novel_classes=False, exact_copy=params['exact_copy'], 
                                   B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])
val_data = generate_icl_gmm_data(gmm, params['val_samples'], params['N'], 
                                 novel_classes=False, exact_copy=params['exact_copy'], 
                                 B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])

train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=params['batch_size'],
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=params['batch_size'],
                       collate_fn=collate_fn)

# Create models
print("\nCreating model...")
wta_model = WinnerTakesAllICL(n_nodes=params['n_nodes'], z_dim=params['D'], 
                          L=params['L'], N=params['N'], activation=params['activation'],
                          sparsity_rho_edge=params['sparsity_rho_edge'], sparsity_rho_all=params['sparsity_rho_all'],
                          f_activation=params['f_activation'], learn_K=params['learn_K'], learn_beta=params['learn_beta'],
                          use_annealing=params['use_annealing'], tau_start=params['tau_start'], tau_end=params['tau_end'],
                          beta_softplus_start=params['beta_softplus_start'], beta_softplus_end=params['beta_softplus_end'], anneal_epochs=params['anneal_epochs'],
                          R0=params['R0'], beta_softplus=params['beta_softplus'])
models_to_train = {'wta': wta_model}

if params['add_qk']:
    qk_model = QKICL(
        z_dim=params['D'],
        L=params['L'],
        N=params['N'],
        d_k=params['qk_dim']
    )
    models_to_train['qk'] = qk_model

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
    unique_labels=params['unique_labels']
)
                     
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Test
results = {}
for name, model in models_to_train.items():
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    results[name] = test_icl(model, gmm, params['N'], device, n_samples=1000,
                             exact_copy=params['exact_copy'], B=params['B'],
                             method=params['method'], L=params['L'],
                             temperature=params['temperature'], shuffle_context=params['shuffle_context'],
                             min_max_choice=params['min_max_choice'], unique_labels=params['unique_labels'])

# WTA-only runs keep the previous flat history/results layout
if not params['add_qk']:
    results = results['wta']
    history = history['wta']

# Save results
os.makedirs(output_dir, exist_ok=True)

for name, model in models_to_train.items():
    if name == 'wta':
        model_path = f'{output_dir}/model.pt'
    else:
        model_path = f'{output_dir}/model_{name}.pt'
    torch.save(model.state_dict(), model_path)

results_data = {
    'results': results,
    'history': history,
    'params': params,
    'execution_time': end_time - start_time
}
results_path = f'{output_dir}/results.pkl'
with open(results_path, "wb") as file:
    pickle.dump(results_data, file)

print("")
for name in models_to_train.keys():
    if name == 'wta':
        print(f"✓ Saved {name.upper()} model to {output_dir}/model.pt")
    else:
        print(f"✓ Saved {name.upper()} model to {output_dir}/model_{name}.pt")
print(f"✓ Saved results to {results_path}")
print(f"✓ Execution Time: {end_time - start_time:.2f} seconds")
