"""
Enzyme-cascade ICL model (graph-forward path).

A layered feed-forward signaling cascade (as in the GlycanAnalysis CRN code):
receptors R_i -> hidden layers of substrates S -> output layer of S nodes.
Each substrate has an inactive form (S_s) and an active form (S) with a fixed
conservation total l0. The steady-state active concentration of node i is

    a_i = l0_i * (kf_i + sum_p kf_ip a_p) / ((kf_i + kr_i) + sum_p (kf_ip + kr_ip) a_p)

where p runs over predecessors (receptors for the first layer, previous layer
otherwise). This is the closed-form recursion of ``GraphComputation.forward``,
vectorised over the batch and per layer in torch.

ICL wiring (mirrors the MLP baseline):
    z_flat (N+1)*D -> receptor concentrations exp(input_scale * z)   [exp scaling]
    -> cascade -> N output-node concentrations q   (n_phos_states=3: the active form is the
       doubly phosphorylated S2 of an ordered S0 <-> S1 <-> S2 chain)
    -> softmax(q / temperature) = attention over context positions
    -> aggregate attention mass by context label -> log-probs over L classes.

Trainable pieces can be selected independently:
    learn_rates: train log kf/kr (node + edge); otherwise frozen at random init
    learn_l0:    train log l0 per node (receptors are driven by the input)
"""

import numpy as np
import torch
import torch.nn as nn

from .base_icl_model import BaseICLModel

_EPS = 1e-10


class EnzymeCascadeICL(BaseICLModel):
    def __init__(
        self,
        z_dim=2,
        L=75,
        N=4,
        hidden_dim=4,
        hidden_depth=1,
        p_f=1.0,
        input_scale=1.0,
        exp_input=True,
        input_clamp=20.0,
        rate_init_std=1.0,
        l0_init=1.0,
        learn_rates=True,
        learn_l0=True,
        structure_seed=None,
        print_creation=True,
        n_phos_states=2,
        rate_anchor_lambda=0.0,
        rate_init_mode="random",
        edge_strength=1.0,
        p_activating=0.5,
        signed_off_ratio=1e-3,
        signed_node_rate=1.0,
    ):
        """
        Args:
            z_dim, L, N: as in other ICL models. Receptors = (N+1)*z_dim.
            hidden_dim, hidden_depth: hidden layers of the cascade. The output
                layer always has N nodes (one per context element).
            p_f: probability of each forward connection (receptor->layer 0 and
                layer->layer). Every receptor gets at least one target in layer 0.
            input_scale: receptor conc = exp(input_scale * z), z clamped to +-input_clamp.
            exp_input: if False, receptor conc = input_scale * z (clamped, no exp).
            rate_init_std: rates initialised as exp(randn * std) (as in run_training.py).
            l0_init: initial conservation total for every substrate node.
            learn_rates: if False, rates are frozen at their random init.
            learn_l0: if False, l0 is frozen.
            structure_seed: seed for the random connectivity (uses a local RNG).
            n_phos_states: 2 (default) = inactive/active substrate. 3 = ordered
                chain S0 <-> S1 <-> S2 (unphosphorylated, singly, doubly
                phosphorylated); the active form is S2. Step 1 uses the
                original kf/kr parameters, step 2 has its own kf2/kr2 set
                (node + edge), drawn after all step-1 params so the 2-state
                init is unchanged for a given seed.
            rate_anchor_lambda: strength of the anchor penalty that keeps the
                (log) kinetic rates close to their initial values:
                    lambda * mean_over_active_rates (log k - log k_init)^2
                Only active edges (mask == 1) count. Applied by the training
                loop through ``rate_anchor_penalty()``; has no effect when
                learn_rates=False. l0 is never penalised.
            rate_init_mode: "random" (default; log-normal rates with rate_init_std) or
                "signed". In "signed" mode every edge i -> j gets a fixed strength and
                a random sign: an *activating* edge has kf_edge = edge_strength and
                kr_edge = edge_strength * signed_off_ratio; an *inactivating* edge is
                the reverse. Node (basal) rates are all signed_node_rate. rate_init_std
                is ignored. The 3-state second step reuses the same signs.
            edge_strength: kinetic strength of the 'on' channel of each edge (signed mode).
            p_activating: probability an edge is activating (signed mode).
            signed_off_ratio: 'off' channel strength relative to 'on' (signed mode).
                Must be > 0 because rates are stored as logs; 1e-3 is ~ "absent".
            signed_node_rate: basal kf_node = kr_node value (signed mode).
        """
        super().__init__(n_nodes=None, z_dim=z_dim, L=L, N=N)
        if hidden_depth < 0:
            raise ValueError("hidden_depth must be >= 0")

        if n_phos_states not in (2, 3):
            raise ValueError("n_phos_states must be 2 or 3")
        self.n_phos_states = n_phos_states

        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.p_f = p_f
        self.input_scale = input_scale
        self.exp_input = exp_input
        self.input_clamp = input_clamp
        self.learn_rates = learn_rates
        self.learn_l0 = learn_l0

        self.n_receptors = (N + 1) * z_dim
        self.layer_sizes = [hidden_dim] * hidden_depth + [N]
        self.n_layers = len(self.layer_sizes)

        rng = np.random.RandomState(structure_seed)
        prev_sizes = [self.n_receptors] + self.layer_sizes[:-1]

        # Connectivity masks: mask[l] has shape (n_l, n_prev_l); entry [j, i] = edge i -> j.
        masks = []
        for l, (n_l, n_prev) in enumerate(zip(self.layer_sizes, prev_sizes)):
            m = (rng.rand(n_l, n_prev) < p_f).astype(np.float32)
            if l == 0:
                # Every receptor must feed at least one first-layer substrate.
                for i in range(n_prev):
                    if m[:, i].sum() == 0:
                        m[rng.randint(n_l), i] = 1.0
            masks.append(m)

        def rate_param(shape):
            return nn.Parameter(
                torch.from_numpy(rng.randn(*shape).astype(np.float32) * rate_init_std),
                requires_grad=learn_rates,
            )

        # ---- optional signed-interaction init (drawn after the masks, so the
        # connectivity is identical to the 'random' mode for a given seed) ----
        if rate_init_mode not in ("random", "signed"):
            raise ValueError("rate_init_mode must be 'random' or 'signed'")
        signed = rate_init_mode == "signed"
        if signed:
            # activating[l][j, i] = True: edge i -> j promotes activation of j
            # (contributes to kf); False: it promotes inactivation (contributes to kr).
            activating = [
                rng.rand(n_l, n_prev) < p_activating
                for n_l, n_prev in zip(self.layer_sizes, prev_sizes)
            ]
            log_on = float(np.log(edge_strength))
            log_off = float(np.log(edge_strength * signed_off_ratio))
            log_node = float(np.log(signed_node_rate))
            self.signed_activating = [a.copy() for a in activating]  # for inspection

        def node_p(shape):
            if not signed:
                return rate_param(shape)
            return nn.Parameter(torch.full(shape, log_node), requires_grad=learn_rates)

        def edge_p(l, kind):
            """kind 'f' (activating channel) or 'r' (inactivating channel)."""
            shape = masks[l].shape
            if not signed:
                return rate_param(shape)
            act = activating[l] if kind == "f" else ~activating[l]
            vals = np.where(act, log_on, log_off).astype(np.float32)
            return nn.Parameter(torch.from_numpy(vals), requires_grad=learn_rates)

        self.masks = nn.ParameterList()  # non-trainable, saved in state_dict
        self.log_kf_node = nn.ParameterList()
        self.log_kr_node = nn.ParameterList()
        self.log_kf_edge = nn.ParameterList()
        self.log_kr_edge = nn.ParameterList()
        self.log_l0 = nn.ParameterList()
        for l, (n_l, n_prev) in enumerate(zip(self.layer_sizes, prev_sizes)):
            self.masks.append(nn.Parameter(torch.from_numpy(masks[l]), requires_grad=False))
            self.log_kf_node.append(node_p((n_l,)))
            self.log_kr_node.append(node_p((n_l,)))
            self.log_kf_edge.append(edge_p(l, "f"))
            self.log_kr_edge.append(edge_p(l, "r"))
            self.log_l0.append(
                nn.Parameter(
                    torch.full((n_l,), float(np.log(l0_init))),
                    requires_grad=learn_l0,
                )
            )

        # Second phosphorylation step (S1 <-> S2). Drawn after all step-1 params.
        if n_phos_states == 3:
            self.log_kf2_node = nn.ParameterList()
            self.log_kr2_node = nn.ParameterList()
            self.log_kf2_edge = nn.ParameterList()
            self.log_kr2_edge = nn.ParameterList()
            for l, (n_l, n_prev) in enumerate(zip(self.layer_sizes, prev_sizes)):
                self.log_kf2_node.append(node_p((n_l,)))
                self.log_kr2_node.append(node_p((n_l,)))
                self.log_kf2_edge.append(edge_p(l, "f"))  # same sign as step 1
                self.log_kr2_edge.append(edge_p(l, "r"))

        # Anchor targets: copies of the initial log-rates (non-persistent buffers so
        # old checkpoints still load and the anchors follow .to(device)/.double()).
        self.rate_anchor_lambda = float(rate_anchor_lambda)
        for name in self._rate_param_names():
            for l, p in enumerate(getattr(self, name)):
                self.register_buffer(f"{name}_{l}_init0", p.detach().clone(), persistent=False)

        if print_creation:
            n_edges = int(sum(m.sum() for m in masks))
            print(
                f"  Initialized Enzyme Cascade ICL (graph forward): receptors={self.n_receptors}, "
                f"layers={self.layer_sizes}, edges={n_edges}, p_f={p_f}, "
                f"phos_states={n_phos_states}"
            )
            print(
                f"  input_scale={input_scale}, exp_input={exp_input}, learn_rates={learn_rates}, learn_l0={learn_l0}, "
                f"trainable params: {self.get_num_parameters():,}"
            )

    def _rate_param_names(self):
        names = ["log_kf_node", "log_kr_node", "log_kf_edge", "log_kr_edge"]
        if self.n_phos_states == 3:
            names += ["log_kf2_node", "log_kr2_node", "log_kf2_edge", "log_kr2_edge"]
        return names

    def _rate_drift_sums(self):
        """(sum sq drift, count, max |drift|, sum |drift|) of log-rates over active rates."""
        total, count, dmax, abs_total = 0.0, 0.0, 0.0, 0.0
        for name in self._rate_param_names():
            for l, p in enumerate(getattr(self, name)):
                d2 = (p - getattr(self, f"{name}_{l}_init0")) ** 2
                if name.endswith("_edge"):
                    w = self.masks[l]
                    d2 = d2 * w
                    count = count + w.sum()
                else:
                    count = count + d2.numel()
                total = total + d2.sum()
                abs_total = abs_total + d2.detach().sqrt().sum()
                dmax = max(dmax, float(d2.max().detach()) ** 0.5)
        return total, count, dmax, abs_total

    def rate_anchor_penalty(self):
        """lambda * mean (log k - log k_init)^2 over active rates (0 if disabled/frozen)."""
        if self.rate_anchor_lambda == 0.0 or not self.learn_rates:
            return self.log_l0[0].sum().detach() * 0.0
        total, count, _, _ = self._rate_drift_sums()
        return self.rate_anchor_lambda * total / count

    @torch.no_grad()
    def rate_drift(self):
        """Diagnostics over active rates of |log k - log k_init|: mean_abs, rms, max.

        mean_abs = average amount each rate has moved (in log units; 1.0 = a factor of e).
        """
        total, count, dmax, abs_total = self._rate_drift_sums()
        return {
            "mean_abs": float(abs_total / count),
            "rms": float((total / count) ** 0.5),
            "max": dmax,
        }

    def receptor_concentrations(self, z_flat):
        """Map the ICL input into receptor concentrations (exp or linear scaling)."""
        x = torch.clamp(self.input_scale * z_flat, min=-self.input_clamp, max=self.input_clamp)
        return torch.exp(x) if self.exp_input else x

    def cascade(self, receptors):
        """Graph forward. receptors: (B, n_receptors) -> list of per-layer active conc (B, n_l)."""
        a_prev = receptors
        outputs = []
        for l in range(self.n_layers):
            mask = self.masks[l]
            kf_n = torch.exp(self.log_kf_node[l])
            kr_n = torch.exp(self.log_kr_node[l])
            kf_e = torch.exp(self.log_kf_edge[l]) * mask  # (n_l, n_prev)
            kr_e = torch.exp(self.log_kr_edge[l]) * mask
            l0 = torch.exp(self.log_l0[l])

            u1 = kf_n + a_prev @ kf_e.T                       # (B, n_l) S0 -> S1
            d1 = kr_n + a_prev @ kr_e.T                       # S1 -> S0

            if self.n_phos_states == 2:
                a = l0 * u1 / (u1 + d1 + _EPS)
            else:
                kf2_n = torch.exp(self.log_kf2_node[l])
                kr2_n = torch.exp(self.log_kr2_node[l])
                kf2_e = torch.exp(self.log_kf2_edge[l]) * mask
                kr2_e = torch.exp(self.log_kr2_edge[l]) * mask
                u2 = kf2_n + a_prev @ kf2_e.T                 # S1 -> S2
                d2 = kr2_n + a_prev @ kr2_e.T                 # S2 -> S1
                # Birth-death steady state: pi ~ (1, u1/d1, u1*u2/(d1*d2)).
                # Active fraction = pi2 / sum(pi), multiplied through by d1*d2.
                a = l0 * (u1 * u2) / (d1 * d2 + u1 * d2 + u1 * u2 + _EPS)
            outputs.append(a)
            a_prev = a
        return outputs

    def forward(self, z_seq_batch, labels_seq_batch, method=None, temperature=1.0):
        """
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N), labels in 1..L
            method: unused (interface compatibility)
            temperature: softmax temperature over output-node concentrations

        Returns:
            log_probs: (batch_size, L)
        """
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)
        q = self.cascade(self.receptor_concentrations(z_flat))[-1]  # (B, N)
        attention = torch.softmax(q / temperature, dim=1)

        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1, num_classes=self.L
        ).to(attention.dtype)
        class_probs = torch.einsum("bn,bnl->bl", attention, labels_one_hot)
        class_probs = class_probs.clamp(min=1e-6, max=1.0)
        return torch.log(class_probs)
