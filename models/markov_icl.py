"""
Markov ICL Model using Matrix Tree Theorem.

This implementation uses a rate matrix W computed from parameters K where:
- W_ij ≥ 0 for i ≠ j (transition rates from state j to state i)
- Columns sum to zero: Σᵢ W_ij = 0
- Steady state satisfies: W p = 0

The Matrix Tree Theorem gives:
    p_i = det(W^(i)) / Σⱼ det(W^(j))
    
where W^(i) is obtained by deleting row i and column i from W.

Naming convention:
- Linear rate encoder: K_params maps flattened z to per-edge log-rate shifts.
- MLP rate encoder: MLP maps z to n_nodes^2 shifts (full connectivity; ignores K sparsity masks).
- W: computed rate matrix from base rates, encoder modulation, and optional label mod.
- Rate decoder: maps steady-state π to length-N scores (canonical name); `context_scorer*` kept as aliases.
"""

import os
import warnings
import torch
import torch.nn as nn
import numpy as np
from .base_icl_model import BaseICLModel

_LEGACY_DECODER_PREFIX = "context_scorer."
_CANON_DECODER_PREFIX = "rate_decoder."

# If True, hidden_map='linear' sums off-diagonal transform(M) instead of M (log-rates).
LINEAR_HIDDEN_MAP_APPLY_TRANSFORM = True
# If True, hidden_map='frozen_mlp' feeds off-diagonal transform(M) into the MLP; else raw M.
FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM = True


LOG_RATE_CLAMP_MIN = float(np.log(1e-8))
LOG_RATE_CLAMP_MAX = float(np.log(1e8))


class MatrixTreeMarkovICL(BaseICLModel):
    """
    Matrix Tree Theorem implementation using rate matrix W with CLASSIFICATION output.
    
    Architecture:
    1. Compute context-dependent rate matrix W (from parameters K)
    2. Solve for steady state distribution π: W p = 0
    3. Map π to attention over context positions
    4. Aggregate attention by label to get class logits
    """
    
    def __init__(self, n_nodes=10, z_dim=2, L=75, N=4, use_label_mod=False,
                 learn_base_rates=True, transform_func='exp',
                 sparsity_rho_edge=1.0, sparsity_rho_all=1.0,
                 sparsity_rho_edge_base_W=1.0, base_mask_value=0.0,
                 rate_encoder_type='linear',
                 encoder_mlp_depth=2,
                 encoder_mlp_width=64,
                 rate_decoder_type=None,
                 decoder_mlp_depth=None,
                 decoder_mlp_width=None,
                 context_scorer_type='linear',
                 mlp_depth=None,
                 mlp_width=None,
                 print_creation=True,
                 rate_param='directed',
                 db_phi0=0.0,
                 db_lambda=100.0,
                 hidden_map='crn',
                 hidden_mlp_depth=2,
                 hidden_mlp_width=64,
                 learn_rate_encoder=True,
                 learn_rate_decoder=True):
        """
        Initialize Markov ICL model.
        
        Args:
            n_nodes: Number of Markov chain nodes
            z_dim: Dimension of input features
            L: Number of output classes
            N: Number of context examples
            use_label_mod: Whether to modulate rates by context labels
            learn_base_rates: Whether to allow gradient updates to unmasked base rates
            transform_func: Transformation function for rates
                ('exp', 'relu', 'elu', 'softplus', 'sigmoid', 'identity')
            sparsity_rho_edge: Fraction of non-zero elements in per-edge mask (n_nodes x n_nodes)
            sparsity_rho_all: Fraction of non-zero elements in per-element mask (all dims)
            sparsity_rho_edge_base_W: Fraction of (i,j) edges with base rates in W
            base_mask_value: Value for masked base rates (0.0 or float('-inf'))
            rate_encoder_type: 'linear' (K_params · z) or 'mlp' (dense z → n_nodes² shift)
            encoder_mlp_depth: MLP layers for rate encoder (>=2) when rate_encoder_type='mlp'
            encoder_mlp_width: Hidden width for rate encoder MLP
            rate_decoder_type: 'linear' or 'mlp' (π → context scores). Preferred over context_scorer_type.
            decoder_mlp_depth: MLP depth for rate decoder (>=2) when rate_decoder_type='mlp'
            decoder_mlp_width: Hidden width for rate decoder MLP
            context_scorer_type: Deprecated alias for rate_decoder_type (must agree if both set)
            mlp_depth: Deprecated alias for decoder_mlp_depth
            mlp_width: Deprecated alias for decoder_mlp_width
            rate_param: 'directed' (original independent log-rates) or 'db_split'
                (node energy / symmetric barrier / antisymmetric force).
            db_phi0: Hinge threshold on each force scalar (only used if rate_param='db_split').
            db_lambda: Weight on the detailed-balance hinge penalty.
            hidden_map: How M maps to the n-vector that hits the decoder:
                'crn' (steady state of W), 'linear' (off-diagonal column sums of M,
                or of transform(M) if LINEAR_HIDDEN_MAP_APPLY_TRANSFORM),
                'relu' (ReLU of those sums of M), or 'frozen_mlp' (frozen MLP on
                off-diagonal transform(M) if FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM,
                else off-diagonal M).
            hidden_mlp_depth: MLP layers (>=2) when hidden_map='frozen_mlp'.
            hidden_mlp_width: Hidden width for the frozen rate→π MLP.
            learn_rate_encoder: If False, freeze K / base (or DB tensors) at init.
            learn_rate_decoder: If False, freeze B / MLP decoder at a semi-orthogonal init.
        """
        super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
        self.n_nodes = n_nodes
        self.use_label_mod = use_label_mod
        self.transform_func = transform_func
        self.sparsity_rho_edge = sparsity_rho_edge
        self.sparsity_rho_all = sparsity_rho_all
        self.sparsity_rho_edge_base_W = sparsity_rho_edge_base_W
        self.base_mask_value = base_mask_value
        self.learn_base_rates = learn_base_rates
        if rate_param not in ('directed', 'db_split'):
            raise ValueError(f"Invalid rate_param: {rate_param}. Expected 'directed' or 'db_split'")
        self.rate_param = rate_param
        self.db_phi0 = float(db_phi0)
        self.db_lambda = float(db_lambda)
        if hidden_map not in ('crn', 'linear', 'relu', 'frozen_mlp'):
            raise ValueError(
                f"Invalid hidden_map: {hidden_map}. "
                "Expected 'crn', 'linear', 'relu', or 'frozen_mlp'"
            )
        self.hidden_map = hidden_map
        self.hidden_mlp_depth = hidden_mlp_depth
        self.hidden_mlp_width = hidden_mlp_width
        self.learn_rate_encoder = bool(learn_rate_encoder)
        self.learn_rate_decoder = bool(learn_rate_decoder)
        if not self.learn_rate_encoder and not self.learn_rate_decoder:
            raise ValueError("Cannot freeze both rate encoder and rate decoder")

        if rate_decoder_type is not None and context_scorer_type != 'linear':
            if rate_decoder_type != context_scorer_type:
                raise ValueError(
                    "Conflicting rate_decoder_type and context_scorer_type "
                    f"({rate_decoder_type!r} vs {context_scorer_type!r})"
                )
        if rate_decoder_type is not None:
            rd_type = rate_decoder_type
        else:
            rd_type = context_scorer_type
            if context_scorer_type == 'mlp':
                warnings.warn(
                    "context_scorer_type='mlp' is deprecated; use rate_decoder_type='mlp' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if (
            decoder_mlp_depth is not None
            and mlp_depth is not None
            and decoder_mlp_depth != mlp_depth
        ):
            raise ValueError(
                "Conflicting decoder_mlp_depth and mlp_depth "
                f"({decoder_mlp_depth} vs {mlp_depth})"
            )
        if (
            decoder_mlp_width is not None
            and mlp_width is not None
            and decoder_mlp_width != mlp_width
        ):
            raise ValueError(
                "Conflicting decoder_mlp_width and mlp_width "
                f"({decoder_mlp_width} vs {mlp_width})"
            )
        legacy_mlp_depth = mlp_depth if mlp_depth is not None else 2
        legacy_mlp_width = mlp_width if mlp_width is not None else 64
        dd = decoder_mlp_depth if decoder_mlp_depth is not None else legacy_mlp_depth
        dw = decoder_mlp_width if decoder_mlp_width is not None else legacy_mlp_width

        if rate_encoder_type not in ('linear', 'mlp'):
            raise ValueError(f"Invalid rate_encoder_type: {rate_encoder_type}")
        if rate_param == 'db_split':
            if rate_encoder_type != 'linear':
                raise ValueError("rate_param='db_split' requires rate_encoder_type='linear'")
            if transform_func != 'exp' and hidden_map == 'crn':
                raise ValueError(
                    "rate_param='db_split' requires transform_func='exp' when hidden_map='crn'"
                )
            if use_label_mod:
                raise ValueError("rate_param='db_split' does not support use_label_mod=True")
        if rd_type not in ('linear', 'mlp'):
            raise ValueError(
                f"Invalid rate_decoder_type: {rd_type}. Expected 'linear' or 'mlp'"
            )
        if rd_type == 'mlp' and dd < 2:
            raise ValueError("decoder_mlp_depth must be >= 2 when rate_decoder_type='mlp'")
        if rate_encoder_type == 'mlp' and encoder_mlp_depth < 2:
            raise ValueError("encoder_mlp_depth must be >= 2 when rate_encoder_type='mlp'")
        if hidden_map == 'frozen_mlp' and hidden_mlp_depth < 2:
            raise ValueError("hidden_mlp_depth must be >= 2 when hidden_map='frozen_mlp'")

        self.rate_encoder_type = rate_encoder_type
        self.encoder_mlp_depth = encoder_mlp_depth
        self.encoder_mlp_width = encoder_mlp_width
        self.rate_decoder_type = rd_type
        self.decoder_mlp_depth = dd
        self.decoder_mlp_width = dw
        # Deprecated mirrors (kept in sync for external code)
        self.context_scorer_type = rd_type
        self.mlp_depth = dd
        self.mlp_width = dw
        
        z_full_dim = (N + 1) * z_dim  # Flatten all context + query
        self.z_full_dim = z_full_dim
        l_full_dim = N
        
        # Initialize parameters with proper scaling
        init_scale_K = 0.05 / np.sqrt(n_nodes)
        init_scale_B = 0.1 / np.sqrt(N)
        init_base = -2.0 - 0.5 * np.log(n_nodes)
        
        self.energy_bias = None
        self.energy_weight = None
        self.barrier_bias = None
        self.barrier_weight = None
        self.force_bias = None
        self.force_weight = None

        if rate_param == 'db_split':
            self.K_params = None
            self.rate_encoder = None
            self.base_log_rates_W = None
            # Node energy E_n(z) = beta_n + kappa_n · z
            self.energy_bias = nn.Parameter(torch.zeros(n_nodes))
            self.energy_weight = nn.Parameter(
                torch.randn(n_nodes, z_full_dim) * init_scale_K
            )
            # Symmetric barrier B_ji(z); only upper triangle (j > i) is used
            self.barrier_bias = nn.Parameter(
                torch.zeros(n_nodes, n_nodes) + (-init_base)
            )
            self.barrier_weight = nn.Parameter(
                torch.randn(n_nodes, n_nodes, z_full_dim) * init_scale_K
            )
            # Antisymmetric force F_ji(z); only upper triangle is used
            self.force_bias = nn.Parameter(torch.zeros(n_nodes, n_nodes))
            self.force_weight = nn.Parameter(
                torch.zeros(n_nodes, n_nodes, z_full_dim)
            )
        elif rate_encoder_type == 'linear':
            self.K_params = nn.Parameter(
                torch.randn(n_nodes, n_nodes, z_full_dim) * init_scale_K
            )
            self.rate_encoder = None
        else:
            self.K_params = None
            out_dim = n_nodes * n_nodes
            ed, ew = encoder_mlp_depth, encoder_mlp_width
            enc_layers = [nn.Linear(z_full_dim, ew), nn.ReLU()]
            for _ in range(ed - 2):
                enc_layers.extend([nn.Linear(ew, ew), nn.ReLU()])
            enc_layers.append(nn.Linear(ew, out_dim))
            self.rate_encoder = nn.Sequential(*enc_layers)
        
        # Optional: modulate rates by context labels
        if self.use_label_mod:
            self.label_modulation = nn.Parameter(
                torch.randn(n_nodes, n_nodes, l_full_dim) * init_scale_K * 0.5
            )
        else:
            self.label_modulation = None
        
        # Rate decoder: node features → context position scores
        if rd_type == 'linear':
            if self.learn_rate_decoder:
                self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)
            else:
                self.B = nn.Parameter(self._semi_orthogonal_decoder(n_nodes, N))
                self.B.requires_grad = False
            self.rate_decoder = None
        else:
            layers = [nn.Linear(n_nodes, dw), nn.ReLU()]
            for _ in range(dd - 2):
                layers.extend([nn.Linear(dw, dw), nn.ReLU()])
            layers.append(nn.Linear(dw, N))
            self.rate_decoder = nn.Sequential(*layers)
            self.B = None
            if not self.learn_rate_decoder:
                for p in self.rate_decoder.parameters():
                    p.requires_grad = False
        self.context_scorer = self.rate_decoder

        # Frozen MLP: off-diagonal M (or transform(M)) → n-vector (random features, not trained).
        self.hidden_mlp = None
        if hidden_map == 'frozen_mlp':
            in_dim = n_nodes * (n_nodes - 1)
            hd, hw = hidden_mlp_depth, hidden_mlp_width
            hidden_layers = [nn.Linear(in_dim, hw), nn.ReLU()]
            for _ in range(hd - 2):
                hidden_layers.extend([nn.Linear(hw, hw), nn.ReLU()])
            hidden_layers.append(nn.Linear(hw, n_nodes))
            self.hidden_mlp = nn.Sequential(*hidden_layers)
            for p in self.hidden_mlp.parameters():
                p.requires_grad = False
        
        if rate_param != 'db_split':
            # Base log rates for W
            # Note: To get zero base rates, set sparsity_rho_edge_base_W = 0.0 with base_mask_value = 0.0
            self.base_log_rates_W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1 + init_base)

            # Create sparsity masks for K_params (linear encoder) and base rates
            self._create_sparsity_masks(z_full_dim)

            # Set up gradient masking for base rates if learn_base_rates is False
            # or if we want to only learn unmasked base rates
            if not learn_base_rates:
                self.base_log_rates_W.requires_grad = False
            else:
                # Register hooks to zero out gradients for masked base rates
                self.base_log_rates_W.register_hook(
                    lambda grad: grad * self.base_log_rates_W_mask
                )

        if not self.learn_rate_encoder:
            self._freeze_rate_encoder()
        
        if print_creation:
            print(f"  Initialized ICL Attention model (L={L} classes, "
                f"attention over {N} context items)")
            print(f"  Label modulation: {self.use_label_mod}")
            print(f"  Hidden map: {hidden_map}")
            if hidden_map == 'linear':
                if LINEAR_HIDDEN_MAP_APPLY_TRANSFORM:
                    print(
                        f"  Linear hidden map: column sums of {transform_func}(M) "
                        f"(LINEAR_HIDDEN_MAP_APPLY_TRANSFORM={LINEAR_HIDDEN_MAP_APPLY_TRANSFORM})"
                    )
                else:
                    print(
                        f"  Linear hidden map: column sums of M "
                        f"(LINEAR_HIDDEN_MAP_APPLY_TRANSFORM={LINEAR_HIDDEN_MAP_APPLY_TRANSFORM})"
                    )
            if hidden_map == 'frozen_mlp':
                src = (
                    f"{transform_func}(M)"
                    if FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM
                    else "M"
                )
                print(
                    f"  Frozen hidden MLP: depth={hidden_mlp_depth}, "
                    f"width={hidden_mlp_width}, activation=relu "
                    f"(off-diagonal {src} → π; weights frozen; "
                    f"FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM={FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM})"
                )
            print(f"  Learn rate encoder: {self.learn_rate_encoder}")
            print(f"  Learn rate decoder: {self.learn_rate_decoder}")
            if rate_param == 'db_split':
                print(f"  DB hinge: phi0={self.db_phi0}, lambda={self.db_lambda}")
            else:
                print(f"  Base rates learnable: {learn_base_rates}")
            print(f"  Rate encoder: {rate_encoder_type}")
            if rate_encoder_type == 'mlp':
                print(
                    f"  Encoder MLP: depth={encoder_mlp_depth}, width={encoder_mlp_width}, "
                    "activation=relu"
                )
            print(f"  Rate decoder: {rd_type}")
            if rd_type == 'mlp':
                print(f"  Decoder MLP: depth={dd}, width={dw}, activation=relu")
            if rate_param != 'db_split':
                print(f"  Base mask value: {base_mask_value}")
                print(f"  Sparsity K: rho_edge={sparsity_rho_edge:.3f}, rho_all={sparsity_rho_all:.3f}")
                print(f"  Sparsity base_W: rho_edge={sparsity_rho_edge_base_W:.3f}")
                sparsity_stats = self.get_sparsity_stats()
                if sparsity_stats:
                    print(f"  K_params sparsity: {sparsity_stats['K_actual_sparsity']:.3f} "
                        f"({sparsity_stats['K_num_active']}/{sparsity_stats['K_num_total']} active)")
                    print(f"  base_W sparsity: {sparsity_stats['base_W_actual_sparsity']:.3f} "
                        f"({sparsity_stats['base_W_num_active']}/{sparsity_stats['base_W_num_total']} active)")
            print(f"  Parameters: {self.get_num_parameters():,}")
    
    def _create_sparsity_masks(self, z_full_dim):
        """
        Create sparsity masks for K_params and base rates using two-level masking.
        
        For K_params (n_nodes, n_nodes, z_full_dim):
            - Per-edge mask: (n_nodes, n_nodes, 1) - controls which (i,j) edges exist
            - Per-element mask: (n_nodes, n_nodes, z_full_dim) - controls sparsity within each edge
        
        For base_log_rates_W (n_nodes, n_nodes):
            - Per-edge mask: (n_nodes, n_nodes) - controls which (i,j) edges have base rates
            - IMPORTANT: Automatically set to 1 (enabled) for any edge where K_params is active
              This ensures edges with learnable K parameters aren't permanently disabled by -inf base rates
        
        Final mask is element-wise product: only survives if both masks are 1.
        
        Args:
            z_full_dim: Full dimension of z features
        """
        n = self.n_nodes

        if self.rate_encoder_type == 'mlp':
            # Dense encoder: treat every (i,j) as input-modulated for base-rate mask coupling.
            k_edge_active = torch.ones(n, n)
            if self.sparsity_rho_edge_base_W < 1.0:
                edge_mask_samples_W = torch.rand(n, n)
                base_mask_W = (edge_mask_samples_W < self.sparsity_rho_edge_base_W).float()
            else:
                base_mask_W = torch.ones(n, n)
            base_mask_W = torch.maximum(base_mask_W, k_edge_active)
            self.register_buffer('base_log_rates_W_mask', base_mask_W)
            return
        
        # Per-edge mask: same across all input dimensions
        if self.sparsity_rho_edge < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge
            edge_mask_samples = torch.rand(n, n, 1)
            edge_mask = (edge_mask_samples < self.sparsity_rho_edge).float()
            # Broadcast to full dimension
            edge_mask = edge_mask.expand(-1, -1, z_full_dim).contiguous()
        else:
            edge_mask = torch.ones(n, n, z_full_dim)
        
        # Per-element mask: independent for each element
        if self.sparsity_rho_all < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_all
            element_mask_samples = torch.rand(n, n, z_full_dim)
            element_mask = (element_mask_samples < self.sparsity_rho_all).float()
        else:
            element_mask = torch.ones(n, n, z_full_dim)
        
        # Combine masks: element survives only if both masks are 1
        combined_mask = edge_mask * element_mask
        
        # Register as buffer (moves with model to device, not trained)
        self.register_buffer('K_params_mask', combined_mask)
        
        # ==================== base_log_rates_W mask ====================
        if self.sparsity_rho_edge_base_W < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge_base_W
            edge_mask_samples_W = torch.rand(n, n)
            base_mask_W = (edge_mask_samples_W < self.sparsity_rho_edge_base_W).float()
        else:
            base_mask_W = torch.ones(n, n)
        
        # IMPORTANT: Override base mask for edges where K_params is active
        # If K_params has active parameters for edge (i,j), force base_W mask to 1
        # This ensures -inf base rates don't permanently disable edges that can be learned via K
        k_edge_active = (combined_mask.sum(dim=2) > 0).float()  # (n, n) - 1 if any K param is active
        base_mask_W = torch.maximum(base_mask_W, k_edge_active)
        
        # Register as buffer
        self.register_buffer('base_log_rates_W_mask', base_mask_W)
    
    def _pair_masks(self, device, dtype, extra_dim=False):
        """Upper-triangle mask for undirected pairs j > i."""
        n = self.n_nodes
        triu = torch.triu(torch.ones(n, n, device=device, dtype=dtype), diagonal=1)
        if extra_dim:
            triu = triu.unsqueeze(-1)
        return triu

    def _symmetric_from_upper(self, tensor):
        """Build a symmetric matrix from the strictly upper triangle."""
        extra = tensor.dim() == 3
        upper = tensor * self._pair_masks(tensor.device, tensor.dtype, extra_dim=extra)
        if extra:
            return upper + upper.transpose(0, 1)
        return upper + upper.transpose(0, 1)

    def _antisymmetric_from_upper(self, tensor):
        """Build an antisymmetric matrix from the strictly upper triangle (F_ji at [i,j], j>i)."""
        extra = tensor.dim() == 3
        upper = tensor * self._pair_masks(tensor.device, tensor.dtype, extra_dim=extra)
        if extra:
            return upper - upper.transpose(0, 1)
        return upper - upper.transpose(0, 1)

    def _db_split_log_rates(self, z_batch):
        """
        Affine DB split: log k_{j→i} = E_j - B_ji + F_ji/2.
        Stored at W index [i,j] = k_{j→i}.
        """
        energy = self.energy_bias.unsqueeze(0) + torch.matmul(z_batch, self.energy_weight.T)
        barrier_bias = self._symmetric_from_upper(self.barrier_bias)
        barrier_weight = self._symmetric_from_upper(self.barrier_weight)
        force_bias = self._antisymmetric_from_upper(self.force_bias)
        force_weight = self._antisymmetric_from_upper(self.force_weight)
        barrier = barrier_bias.unsqueeze(0) + torch.einsum('ijd,bd->bij', barrier_weight, z_batch)
        force = force_bias.unsqueeze(0) + torch.einsum('ijd,bd->bij', force_weight, z_batch)
        # log W[b,i,j] = E[b,j] - B[b,i,j] + F[b,i,j]/2
        return energy.unsqueeze(1) - barrier + 0.5 * force

    def detailed_balance_penalty(self, phi0=None, lam=None):
        """
        Equal hinge on every force scalar in the upper triangle.

        L_DB = λ Σ_{j>i} [ ReLU(|φ_ji| - φ⁰)² + Σ_m ReLU(|ψ_ji,m| - φ⁰)² ]
        """
        if self.rate_param != 'db_split' or self.force_bias is None:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.zeros((), device=device, dtype=dtype)

        phi0 = self.db_phi0 if phi0 is None else float(phi0)
        lam = self.db_lambda if lam is None else float(lam)
        if lam == 0.0 or phi0 == float('inf'):
            return self.force_bias.sum() * 0.0

        idx = torch.triu_indices(
            self.n_nodes, self.n_nodes, offset=1, device=self.force_bias.device
        )
        force_scalars = torch.cat([
            self.force_bias[idx[0], idx[1]].reshape(-1),
            self.force_weight[idx[0], idx[1], :].reshape(-1),
        ])
        excess = torch.relu(force_scalars.abs() - phi0)
        return lam * excess.pow(2).sum()

    @staticmethod
    def _semi_orthogonal_decoder(n_nodes, N):
        """Well-conditioned frozen B: orthonormal columns if n>=N, else orthonormal rows."""
        raw = torch.randn(n_nodes, N)
        if n_nodes >= N:
            q, _ = torch.linalg.qr(raw, mode='reduced')
            return q
        q, _ = torch.linalg.qr(raw.T, mode='reduced')
        return q.T

    def _freeze_rate_encoder(self):
        """Freeze couplings that build M(z): K/base, MLP encoder, or DB tensors."""
        for p in (
            self.K_params,
            self.base_log_rates_W,
            self.label_modulation,
            self.energy_bias,
            self.energy_weight,
            self.barrier_bias,
            self.barrier_weight,
            self.force_bias,
            self.force_weight,
        ):
            if p is not None:
                p.requires_grad = False
        if self.rate_encoder is not None:
            for p in self.rate_encoder.parameters():
                p.requires_grad = False

    def compute_log_rate_matrix(self, z_batch, labels_batch=None):
        """Affine log-rate matrix M(z), shape (batch, n_nodes, n_nodes)."""
        batch_size = z_batch.shape[0]
        n = self.n_nodes

        if self.rate_param == 'db_split':
            log_rates = self._db_split_log_rates(z_batch)
        else:
            if self.rate_encoder_type == 'linear':
                K_params_masked = self.K_params * self.K_params_mask
                K_expanded = K_params_masked.unsqueeze(0).expand(batch_size, -1, -1, -1)
                rate_mod = torch.einsum('bijd,bd->bij', K_expanded, z_batch)
            else:
                rate_mod = self.rate_encoder(z_batch).view(batch_size, n, n)

            if self.use_label_mod and labels_batch is not None:
                label_expanded = self.label_modulation.unsqueeze(0).expand(batch_size, -1, -1, -1)
                label_mod = torch.einsum('bijd,bd->bij', label_expanded, labels_batch)
                rate_mod = rate_mod + label_mod

            base_masked = torch.where(
                self.base_log_rates_W_mask.bool(),
                self.base_log_rates_W,
                torch.full_like(self.base_log_rates_W, self.base_mask_value)
            )
            base_expanded = base_masked.unsqueeze(0).expand(batch_size, -1, -1)
            log_rates = base_expanded + rate_mod

        return self._clamp_log_rates(log_rates)

    def _clamp_log_rates(self, log_rates):
        """Clamp log-rates to [LOG_RATE_CLAMP_MIN, LOG_RATE_CLAMP_MAX]."""
        return torch.clamp(log_rates, min=LOG_RATE_CLAMP_MIN, max=LOG_RATE_CLAMP_MAX)

    def _apply_rate_transform(self, log_rates):
        """Map log-rate matrix M to rates via transform_func."""
        if self.transform_func == 'exp':
            return torch.exp(log_rates)
        if self.transform_func == 'relu':
            return torch.relu(log_rates) + 1e-10
        if self.transform_func == 'softplus':
            return torch.nn.functional.softplus(log_rates) + 1e-10
        if self.transform_func == 'sigmoid':
            return 10 * torch.sigmoid(log_rates / 10)
        if self.transform_func == 'elu':
            return torch.nn.functional.elu(log_rates) + 1e-10
        if self.transform_func == 'identity':
            return log_rates
        raise ValueError(f"Invalid transform function: {self.transform_func}")

    def compute_rate_matrix_W(self, z_batch, labels_batch=None):
        """
        Compute the rate matrix W from parameters K where columns sum to zero.
        
        Linear encoder: W[i,j] ∝ transform(base + K[i,j]·z + label_mod).
        MLP encoder: W[i,j] ∝ transform(base + MLP(z)[i,j] + label_mod).
        W[j,j] = -Σ_{k≠j} W[k,j]
        
        Args:
            z_batch: (batch_size, z_full_dim) - flattened input features
            labels_batch: (batch_size, N) - optional context labels for rate modulation
            
        Returns:
            W_batch: (batch_size, n_nodes, n_nodes) - computed rate matrix with columns summing to zero
        """
        batch_size = z_batch.shape[0]
        n = self.n_nodes
        log_rates = self.compute_log_rate_matrix(z_batch, labels_batch=labels_batch)
        rates = self._apply_rate_transform(log_rates)

        # Zero out diagonal (we'll set it later)
        eye = torch.eye(n, device=rates.device).unsqueeze(0)
        rates = rates * (1 - eye)
        
        # Construct W with proper diagonal so columns sum to zero
        # W[j,j] = -Σ_{k≠j} W[k,j]
        col_sums = rates.sum(dim=1)  # Sum over rows
        W_batch = rates - torch.diag_embed(col_sums)
        
        return W_batch
    
    def matrix_tree_steady_state(self, W_batch):
        """
        Compute steady state using Matrix Tree Theorem.
        
        For rate matrix W with columns summing to zero,
        the steady state p satisfying W p = 0 is given by:
        
            p_i = det(W^(i)) / Σⱼ det(W^(j))
        
        where W^(i) is W with row i and column i deleted.
        
        Args:
            W_batch: (batch_size, n_nodes, n_nodes) - rate matrix
            
        Returns:
            p_batch: (batch_size, n_nodes) - steady state distributions
        """
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        
        # Compute determinants of all minors
        p_batch = torch.zeros(batch_size, n, device=device)
        
        for i in range(n):
            # Delete row i and column i
            indices = [j for j in range(n) if j != i]
            W_minor = W_batch[:, indices, :][:, :, indices]
            
            # Compute determinant
            det = torch.det(W_minor)
            det = torch.abs(det)  # Handle numerical sign issues
            det = torch.clamp(det, min=1e-10, max=1e10)
            
            p_batch[:, i] = det
        
        # Normalize
        Z = p_batch.sum(dim=1, keepdim=True)
        Z = torch.clamp(Z, min=1e-8)
        p_batch = p_batch / Z
        
        # Handle NaN/Inf (fallback to uniform)
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / n
        
        return p_batch
    
    def linear_solver_steady_state(self, W_batch):
        """
        Compute steady state using linear solver (more efficient than Matrix Tree).
        
        Solves the augmented system:
            [W      ]     [0]
            [1,1,...] p = [1]
        
        Where W p = 0 (steady state) and sum(p) = 1 (normalization).
        
        Args:
            W_batch: (batch_size, n_nodes, n_nodes) - rate matrix
            
        Returns:
            p_batch: (batch_size, n_nodes)
        """
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        
        # Augment system
        ones_row = torch.ones(batch_size, 1, n, device=device)
        A_augmented = torch.cat([W_batch, ones_row], dim=1)
        
        # Target
        b = torch.zeros(batch_size, n + 1, device=device)
        b[:, -1] = 1.0
        
        # Solve using least squares
        try:
            p_batch = torch.linalg.lstsq(A_augmented, b.unsqueeze(-1)).solution.squeeze(-1)
        except RuntimeError:
            # Fallback: Manual normal equations with regularization
            AtA = torch.bmm(A_augmented.transpose(1, 2), A_augmented)
            AtA = AtA + 1e-6 * torch.eye(n, device=device).unsqueeze(0)
            Atb = torch.bmm(A_augmented.transpose(1, 2), b.unsqueeze(-1))
            p_batch = torch.linalg.solve(AtA, Atb).squeeze(-1)
        
        # Ensure non-negativity and normalization
        p_batch = torch.clamp(p_batch, min=0.0)
        p_batch = p_batch / (p_batch.sum(dim=1, keepdim=True) + 1e-8)
        
        # Handle NaN/Inf
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / self.n_nodes
        
        return p_batch
    
    def _singular_linear_systems(self, A, rcond=1e-12):
        """Boolean mask of batch elements that are singular / too ill-conditioned to solve.

        ``rcond`` is relative to the largest singular value: skip when
        ``smin < rcond * smax``. 1e-6 was too aggressive for typical rate
        matrices (cond ~ 1e6–1e8 is common and still solvable); 1e-12 only
        flags near-machine-singular systems.
        """
        batch_size = A.shape[0]
        device = A.device
        try:
            # svdvals: relative smallest singular value (stable for large n, unlike det).
            if hasattr(torch.linalg, "svdvals"):
                svals = torch.linalg.svdvals(A.detach())
            else:
                _, svals, _ = torch.svd(A.detach())
            smax, _ = svals.max(dim=-1)
            smin, _ = svals.min(dim=-1)
            return smin < (smax.clamp(min=1e-30) * rcond)
        except RuntimeError:
            singular = torch.ones(batch_size, dtype=torch.bool, device=device)
            for i in range(batch_size):
                try:
                    if hasattr(torch.linalg, "svdvals"):
                        torch.linalg.svdvals(A[i].detach())
                    else:
                        torch.svd(A[i].detach())
                    singular[i] = False
                except RuntimeError:
                    singular[i] = True
            return singular

    @staticmethod
    def _diag_ridge(A, ridge=1e-6):
        """Tikhonov ridge scaled by each matrix's max entry: A + (ridge * max|A|) I."""
        n = A.shape[-1]
        scale = A.detach().abs().amax(dim=(-2, -1)).clamp(min=1e-8)
        eye = torch.eye(n, device=A.device, dtype=A.dtype)
        return A + (ridge * scale).view(-1, 1, 1) * eye

    def _add_solve_stats(self, n_ridged=0, n_skipped=0, n_backward=0):
        self._n_ridged = getattr(self, '_n_ridged', 0) + n_ridged
        self._n_skipped = getattr(self, '_n_skipped', 0) + n_skipped
        self._n_backward_skipped = getattr(self, '_n_backward_skipped', 0) + n_backward

    def reset_solve_stats(self):
        self._n_ridged = 0
        self._n_skipped = 0
        self._n_backward_skipped = 0

    def report_solve_stats(self, prefix="", reset=True):
        """Print a one-line ridge/skip summary if anything was counted."""
        n_ridged = getattr(self, '_n_ridged', 0)
        n_skipped = getattr(self, '_n_skipped', 0)
        n_backward = getattr(self, '_n_backward_skipped', 0)
        if n_ridged or n_skipped or n_backward:
            msg = f"{prefix}Solve stats: ridged {n_ridged}, skipped {n_skipped}"
            if n_backward:
                msg += f", backward-skipped {n_backward}"
            print(msg + ".")
        if reset:
            self.reset_solve_stats()

    def _solve_skip_singular(self, A, b, rcond=1e-12, ridge=1e-6):
        """
        Batched solve that ridges ill-conditioned systems, then skips leftovers.

        Near-singular batch elements get ``A + (ridge * max|A|) I`` first so
        they can still contribute to the loss. Only systems that remain
        unsolvable after ridging are replaced by a uniform fallback (no
        gradient) and flagged on ``self._last_solve_valid``. Ridge/skip
        counts are accumulated for ``report_solve_stats``.
        """
        batch_size, n = A.shape[0], A.shape[-1]
        device, dtype = A.device, A.dtype
        p = torch.full((batch_size, n), 1.0 / n, device=device, dtype=dtype)
        singular = self._singular_linear_systems(A, rcond=rcond)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)

        good = ~singular
        if good.any():
            idx_g = torch.nonzero(good, as_tuple=False).squeeze(-1)
            try:
                p.index_copy_(0, idx_g, torch.linalg.solve(A[good], b[good]))
                valid[good] = True
            except RuntimeError:
                singular[good] = True

        if singular.any():
            idx_s = torch.nonzero(singular, as_tuple=False).squeeze(-1)
            n_bad = int(idx_s.numel())
            try:
                p.index_copy_(
                    0, idx_s, torch.linalg.solve(self._diag_ridge(A[singular], ridge), b[singular])
                )
                valid[singular] = True
                self._add_solve_stats(n_ridged=n_bad)
            except RuntimeError:
                n_ridged = 0
                n_skip = 0
                for i in idx_s.tolist():
                    try:
                        p[i] = torch.linalg.solve(
                            self._diag_ridge(A[i:i + 1], ridge), b[i:i + 1]
                        ).squeeze(0)
                        valid[i] = True
                        n_ridged += 1
                    except RuntimeError:
                        n_skip += 1
                self._add_solve_stats(n_ridged=n_ridged, n_skipped=n_skip)

        self._last_solve_valid = valid
        return p

    def direct_solve_steady_state(self, W_batch):
        """
        Replace last row of W with normalization constraint and solve directly.
        
        Args:
            W_batch: (batch_size, n_nodes, n_nodes) - rate matrix
            
        Returns:
            p_batch: (batch_size, n_nodes)
        """
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        
        # Modify W: replace last row with [1, 1, 1, ..., 1]
        W_modified = W_batch.clone()
        W_modified[:, -1, :] = 1.0
        
        # RHS: [0, 0, ..., 0, 1]
        b = torch.zeros(batch_size, n, device=device)
        b[:, -1] = 1.0
        
        p_batch = self._solve_skip_singular(W_modified, b)
        
        # Ensure non-negativity and normalization
        p_batch = torch.clamp(p_batch, min=0.0)
        p_batch = p_batch / (p_batch.sum(dim=1, keepdim=True) + 1e-8)
        
        return p_batch
    
    def newton_steady_state(self, W_batch, n_iter=10, eps=1e-12, tol=1e-8):
        """
        Newton's method for solving W p = 0 with sum(p) = 1.
        
        For the linear steady state problem, this iteratively refines the solution
        using Newton's method with the constraint that probabilities sum to 1.
        
        Args:
            W_batch: (batch_size, n_nodes, n_nodes) - rate matrix
            n_iter: Number of Newton iterations (default: 10)
            eps: Minimum value for clamping probabilities (default: 1e-12)
            tol: Convergence tolerance for ||W p|| (default: 1e-8)
            
        Returns:
            p_batch: (batch_size, n_nodes) - steady state distributions
        """
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        
        # Initialize from uniform distribution
        p_batch = torch.full((batch_size, n), 1.0/n, device=device)
        
        for iteration in range(n_iter):
            # Compute F(p) = W p
            F = torch.bmm(W_batch, p_batch.unsqueeze(-1)).squeeze(-1)  # (batch_size, n)
            
            # Check convergence
            F_norm = torch.abs(F).max()
            if F_norm < tol:
                break
            
            # For linear case, Jacobian J = W
            # We modify the system to enforce sum(p) = 1 by replacing last row
            J_constrained = W_batch.clone()
            J_constrained[:, -1, :] = 1.0
            
            # RHS: -F with constraint correction
            b = -F
            b[:, -1] = 1.0 - p_batch.sum(dim=1)  # Enforce sum constraint
            
            # Solve J * delta_p = b for the Newton step
            try:
                delta_p = torch.linalg.solve(J_constrained, b)
            except RuntimeError:
                # Fallback: add regularization if singular
                J_reg = J_constrained + 1e-6 * torch.eye(n, device=device).unsqueeze(0)
                delta_p = torch.linalg.solve(J_reg, b)
            
            # Update with full Newton step
            p_batch = p_batch + delta_p
            
            # Project onto probability simplex: clamp and normalize
            p_batch = torch.clamp(p_batch, min=eps)
            p_batch = p_batch / p_batch.sum(dim=1, keepdim=True)
        
        # Handle NaN/Inf (fallback to uniform)
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / n
        
        # Verify steady state quality
        with torch.no_grad():
            F_final = torch.bmm(W_batch, p_batch.unsqueeze(-1)).squeeze(-1)
            max_drift = torch.abs(F_final).max().item()
            
            max_drift_tol = 1e-3
            if max_drift > max_drift_tol:
                print("WARNING: Newton steady state did not converge properly!")
                print(f"  Max drift: {max_drift:.2e} (threshold: {max_drift_tol:.2e})")
                print(f"  Iterations: {iteration + 1}/{n_iter}")
        
        return p_batch
    
    def _node_state(self, z_flat, method='direct_solve', labels_batch=None):
        """Map encoder matrix M(z) to an n-vector (π, or a linear/ReLU/frozen-MLP stand-in)."""
        self._last_solve_valid = None
        if self.hidden_map in ('linear', 'relu'):
            M = self.compute_log_rate_matrix(z_flat, labels_batch=labels_batch)
            n = self.n_nodes
            eye = torch.eye(n, device=M.device, dtype=M.dtype).unsqueeze(0)
            rates_or_logs = M
            if self.hidden_map == 'linear' and LINEAR_HIDDEN_MAP_APPLY_TRANSFORM:
                rates_or_logs = self._apply_rate_transform(M)
            p_batch = (rates_or_logs * (1.0 - eye)).sum(dim=1)
            if self.hidden_map == 'relu':
                p_batch = torch.relu(p_batch)
            return p_batch

        if self.hidden_map == 'frozen_mlp':
            M = self.compute_log_rate_matrix(z_flat, labels_batch=labels_batch)
            n = self.n_nodes
            offdiag = ~torch.eye(n, dtype=torch.bool, device=M.device)
            rates_or_logs = M
            if FROZEN_MLP_HIDDEN_MAP_APPLY_TRANSFORM:
                rates_or_logs = self._apply_rate_transform(M)
            k_flat = rates_or_logs[:, offdiag]
            return self.hidden_mlp(k_flat)

        W_batch = self.compute_rate_matrix_W(z_flat, labels_batch=labels_batch)
        if method == 'matrix_tree':
            return self.matrix_tree_steady_state(W_batch)
        if method == 'linear_solver':
            return self.linear_solver_steady_state(W_batch)
        if method == 'newton':
            return self.newton_steady_state(W_batch, n_iter=30)
        if method == 'direct_solve':
            return self.direct_solve_steady_state(W_batch)
        raise ValueError(f"Invalid method: {method}")

    def forward(self, z_seq_batch, labels_seq_batch, method='direct_solve', temperature=1.0):
        """
        Forward pass with attention over context items.
        
        Architecture:
        1. Compute rate matrix W from parameters K
        2. Solve for steady state π: W p = 0
        3. Context position scores: q_m = Σ_k B_{k,m} * π_k
        4. Attention: softmax(q / temperature)
        5. Class logits: sum attention weights by context label
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N) - context labels (1 to L)
            method: str - method for computing steady state
                'matrix_tree', 'linear_solver', 'direct_solve', or 'newton'
            temperature: float - softmax temperature (default 1.0)
            
        Returns:
            logits: (batch_size, L) - class logits (log-probabilities)
        """
        batch_size = z_seq_batch.shape[0]
        device = z_seq_batch.device
        
        # Flatten z sequences
        z_flat = z_seq_batch.reshape(batch_size, -1)
        p_batch = self._node_state(z_flat, method=method)
        
        # Compute context position scores from steady-state probabilities.
        if self.rate_decoder_type == 'linear':
            q = torch.matmul(p_batch, self.B)  # (batch_size, N)
        else:
            q = self.rate_decoder(p_batch)  # (batch_size, N)
        
        # Apply temperature and softmax to get attention over context positions
        attention = torch.softmax(q / temperature, dim=1)  # (batch_size, N)
        
        # Convert context labels to class logits (VECTORIZED)
        # One-hot encode labels: (batch, N) → (batch, N, L)
        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,  # Convert 1-indexed to 0-indexed
            num_classes=self.L
        ).float()
        
        # Aggregate attention weights by label class
        # For each class k, sum attention weights where label = k
        logits = torch.einsum('bn,bnk->bk', attention, labels_one_hot)
        
        # Convert to log-probabilities for NLLLoss
        logits = logits.clamp(min=1e-6, max=1.0)
        logits = torch.log(logits)
        
        return logits

    def load_state_dict(self, state_dict, strict=True):
        """Accept checkpoints saved under legacy name ``context_scorer`` for the decoder MLP."""
        has_new = any(k.startswith(_CANON_DECODER_PREFIX) for k in state_dict)
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith(_LEGACY_DECODER_PREFIX) and not has_new:
                remapped[
                    _CANON_DECODER_PREFIX + key[len(_LEGACY_DECODER_PREFIX):]
                ] = value
            elif key.startswith(_LEGACY_DECODER_PREFIX) and has_new:
                continue
            else:
                remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)
    
    def get_sparsity_stats(self):
        """
        Get statistics about K_params and base rate sparsity.
        
        Returns:
            dict with sparsity information for K and base_W, or None if no masks exist
        """
        if self.rate_param == 'db_split' or not hasattr(self, 'base_log_rates_W_mask'):
            return None
        
        if self.rate_encoder_type == 'linear':
            mask_K = self.K_params_mask
            num_total_K = mask_K.numel()
            num_active_K = mask_K.sum().item()
        else:
            nn = self.n_nodes
            num_total_K = nn * nn
            num_active_K = num_total_K
        actual_sparsity_K = 1.0 - (num_active_K / num_total_K)
        
        # base_log_rates_W stats
        mask_base_W = self.base_log_rates_W_mask
        num_total_base_W = mask_base_W.numel()
        num_active_base_W = mask_base_W.sum().item()
        actual_sparsity_base_W = 1.0 - (num_active_base_W / num_total_base_W)
        
        return {
            'rho_edge': self.sparsity_rho_edge,
            'rho_all': self.sparsity_rho_all,
            'rho_edge_base_W': self.sparsity_rho_edge_base_W,
            'K_actual_sparsity': actual_sparsity_K,
            'K_num_active': int(num_active_K),
            'K_num_total': num_total_K,
            'K_fraction_active': num_active_K / num_total_K,
            'base_W_actual_sparsity': actual_sparsity_base_W,
            'base_W_num_active': int(num_active_base_W),
            'base_W_num_total': num_total_base_W,
            'base_W_fraction_active': num_active_base_W / num_total_base_W
        }
    
    def resample_sparsity_mask(self):
        """
        Re-randomize the sparsity masks with same rho values.
        Useful for experiments testing different random masks.
        Resamples masks for K_params and base_log_rates_W.
        """
        if self.rate_param == 'db_split':
            return
        self._create_sparsity_masks(self.z_full_dim)
        
        # Re-register gradient hooks for base rates if learn_base_rates is True
        if self.learn_base_rates:
            # Remove old hooks (they're automatically replaced when re-registering)
            self.base_log_rates_W.register_hook(
                lambda grad: grad * self.base_log_rates_W_mask
            )
    
    def get_active_edges(self):
        """
        Get list of (i, j) node pairs that have at least one active parameter.
        
        Returns:
            List of tuples [(i, j), ...] representing active edges
        """
        if self.rate_encoder_type == 'mlp':
            return [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes)]
        
        if not hasattr(self, 'K_params_mask'):
            return [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes)]
        
        edge_active = self.K_params_mask.sum(dim=2) > 0  # (n_nodes, n_nodes)
        active_indices = torch.nonzero(edge_active, as_tuple=False)
        
        return [(i.item(), j.item()) for i, j in active_indices]

    def db_violation_measure(self, z_batch=None):
        """
        Mean size of the antisymmetric DB-violation terms.

        Without z_batch: mean |φ̃| and |ψ̃| over upper-triangle scalars.
        With z_batch: mean |F_ji(z)| = |φ̃_ji + ψ̃_ji · z| over samples and pairs.
        Directed models return zeros.
        """
        zeros = {'mean_abs_force_params': 0.0, 'mean_abs_F': None}
        if self.rate_param != 'db_split' or self.force_bias is None:
            return zeros

        idx = torch.triu_indices(
            self.n_nodes, self.n_nodes, offset=1, device=self.force_bias.device
        )
        force_params = torch.cat([
            self.force_bias[idx[0], idx[1]].reshape(-1),
            self.force_weight[idx[0], idx[1], :].reshape(-1),
        ])
        out = {
            'mean_abs_force_params': force_params.abs().mean().item(),
            'mean_abs_F': None,
        }
        if z_batch is None:
            return out

        z_flat = z_batch.reshape(z_batch.shape[0], -1) if z_batch.dim() == 3 else z_batch
        with torch.no_grad():
            Fb = self._antisymmetric_from_upper(self.force_bias)
            Fw = self._antisymmetric_from_upper(self.force_weight)
            F = Fb.unsqueeze(0) + torch.einsum('ijd,bd->bij', Fw, z_flat)
            out['mean_abs_F'] = F[:, idx[0], idx[1]].abs().mean().item()
        return out

    def average_db_violation(self, data_loader, n_samples=500):
        """
        Mean |F(z)| over examples from ``data_loader`` (default 500 samples).

        Also returns the parameter-space mean |φ̃|, |ψ̃|, which does not depend on data.
        """
        out = self.db_violation_measure()
        if self.rate_param != 'db_split' or self.force_bias is None:
            out['mean_abs_F'] = 0.0
            out['n_samples'] = 0
            return out

        device = self.force_bias.device
        weighted_sum = 0.0
        count = 0
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                z_seq = batch[0] if isinstance(batch, (tuple, list)) else batch
                z_seq = z_seq.to(device)
                take = min(n_samples - count, z_seq.shape[0])
                if take <= 0:
                    break
                stats = self.db_violation_measure(z_seq[:take])
                weighted_sum += stats['mean_abs_F'] * take
                count += take
                if count >= n_samples:
                    break

        out['mean_abs_F'] = weighted_sum / count if count else 0.0
        out['n_samples'] = count
        return out

    def get_non_zero_count_K(self):
        if self.K_params is None:
            return 0
        K_array = np.array(self.K_params.detach().numpy() * self.K_params_mask.detach().numpy())
        s = K_array.shape
        non_zero_count = 0
        for i in range(s[0]):
            for j in range(s[1]):
                if i != j: 
                    k_vec = K_array[i,j,:]
                    for element in k_vec:
                        if np.abs(element) > 1e-10:
                            non_zero_count += 1
        return non_zero_count


def load_model(params, path, print_creation = True):
    """Load a MarkovICL model from saved weights.
    
    Args:
        params: Dictionary containing model parameters (as saved by ``run_icl.py``)
        path: Path to directory containing model.pt file (trailing slash optional)
        
    Returns:
        model: Loaded model in evaluation mode on appropriate device
    """
    if not path.endswith(os.sep):
        path = path + os.sep
    z_dim = params.get("input_proj_dim", params["D"])
    model = MatrixTreeMarkovICL(
        n_nodes=params["n_nodes"],
        z_dim=z_dim,
        L=params["L"],
        N=params["N"],
        use_label_mod=params.get("use_label_mod", False),
        learn_base_rates=params.get("learn_base_rates", True),
        transform_func=params.get("transform_func", "exp"),
        sparsity_rho_edge=params.get("sparsity_rho_edge", 1.0),
        sparsity_rho_all=params.get("sparsity_rho_all", 1.0),
        sparsity_rho_edge_base_W=params.get("sparsity_rho_edge_base_W", 1.0),
        base_mask_value=params.get("base_mask_value", 0.0),
        rate_encoder_type=params.get("rate_encoder_type", "linear"),
        encoder_mlp_depth=params.get("encoder_mlp_depth", 2),
        encoder_mlp_width=params.get("encoder_mlp_width", 64),
        rate_decoder_type=params.get("rate_decoder_type"),
        decoder_mlp_depth=params.get("decoder_mlp_depth"),
        decoder_mlp_width=params.get("decoder_mlp_width"),
        context_scorer_type=params.get("context_scorer_type", "linear"),
        mlp_depth=params.get("mlp_depth"),
        mlp_width=params.get("mlp_width"),
        print_creation=print_creation,
        rate_param=params.get("rate_param", "directed"),
        db_phi0=params.get("db_phi0", 0.0),
        db_lambda=params.get("db_lambda", 100.0),
        hidden_map=params.get("hidden_map", "crn"),
        hidden_mlp_depth=params.get("hidden_mlp_depth", 2),
        hidden_mlp_width=params.get("hidden_mlp_width", 64),
        learn_rate_encoder=params.get("learn_rate_encoder", True),
        learn_rate_decoder=params.get("learn_rate_decoder", True),
    )
    
    model_path = path + 'model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    return model
