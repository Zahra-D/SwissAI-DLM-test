# SCION Trace Metrics Reference

`ScionTrace` (`src/discrete_diffusion/optimizers/scion.py`) logs two independent sets of
metrics, gated by two separate config flags in `configs/optim/scion.yaml`. They exist to
empirically estimate the constants `L`, `μ`, `ρ`, `σ*` from
[On the Role of Batch Size in Stochastic Conditional Gradient Methods](https://arxiv.org/abs/2603.21191)
(the "BST" paper), following its Section 6 methodology.

## `stats/*` — per-step trace

**Flag:** `trace_enabled: true`. **Cost:** free — reuses the gradient already computed for
the real training step, no extra forward/backward passes. Logged every optimizer step, from
`ScionTrace.step()`.

| Metric | Formula | Meaning |
|---|---|---|
| `stats/grad_norm_fro_power_1` | `Σ‖g‖²` (all params) | **Squared** Euclidean/Frobenius norm of the current gradient. Note: not square-rooted — it's `‖g‖₂²`, not `‖g‖₂`. The naive gradient-size measure. |
| `stats/grad_norm_nuc_power_1` | `Σ_ℓ ⟨g_ℓ, lmo(g_ℓ)⟩` | The **dual norm** `‖g‖*` of the current gradient (paper eq. 14: sum over layers, each in its own dual norm — nuclear for matrix layers, L1 for sign layers, etc.). **This is the μ input** — regressed against loss to get the μ-KL slope. |
| `stats/num_fro` | `‖Δg‖₂` | Euclidean norm of the gradient *difference* between this step and the last (`Δg = g_{k-1} − g_k`). |
| `stats/den_fro` | `‖Δx‖₂` | Euclidean norm of the weight *difference* between this step and the last. |
| `stats/num_nuc` | `⟨Δg, lmo(Δg)⟩` | Dual norm `‖Δg‖*` of the gradient difference. Numerator for the dual-norm L estimate. |
| `stats/den_spec` | `max_ℓ ‖Δx_ℓ‖_ℓ` | Primal norm `‖Δx‖` of the weight difference (max across layers, per eq. 14). Denominator for the L estimate. |
| `stats/local_smooth_fro` | `num_fro / den_fro` | Euclidean-only curvature proxy — a sanity-check companion, not the quantity the theory actually calls for. |
| `stats/local_smooth_spec` | `num_nuc / den_spec` | **This is the L estimate** — `‖Δg‖*/‖Δx‖`, dual-norm smoothness, matching Section 6.4's methodology exactly. |
| `stats/stats_step` | `self.n_steps` | Optimizer step counter this row corresponds to (useful for joining against `trainer/global_step` if they ever drift). |

`stats/ddp_trace_world_size` and `stats/ddp_trace_*_max_abs_diff` are defined in the code but
currently commented out (they were a diagnostic added to verify cross-rank gradient
consistency; disabled once confirmed).

## `grad/*` and `rho/*` — noise trace

**Flags:** `trace_collect_noise_stats: true`, `trace_noise_stats_every`, `trace_m`,
`trace_noise_min_samples`. **Cost:** not free — every `trace_noise_stats_every` steps, does
`trace_m` extra forward/backward passes on freshly-sampled batches with weights frozen, via
`_collect_scion_trace_m_stats` (`base.py`). All metrics below come from that one batch of
`m = trace_m` independent gradient samples collected at a single frozen point `x_k`, via
`report_stats()`.

| Metric | Formula | Meaning |
|---|---|---|
| `grad/noise_samples` | `m` | How many of the `trace_m` samples were actually collected this firing. |
| `grad/noise_E_grad_norm2` | `Eg2 = (1/m)Σ‖gᵢ‖²` | Average squared norm across the `m` samples — includes both true signal and noise. |
| `grad/noise_mean_grad_norm2` | `mean_g2 = ‖ḡ‖²`, `ḡ=(1/m)Σgᵢ` | Squared norm of the sample mean — a proxy for `‖∇f(x)‖²`, the true (noise-free) gradient. |
| `grad/noise_sigma2` | `(Eg2 − mean_g2) · m/(m−1)` | **σ²** — the bias-variance identity gives the biased variance for free; `m/(m−1)` is Bessel's correction for an unbiased estimate. This is σ² at whatever batch size each sample used (≈ `global_batch_size`, post gradient-sync fix). |
| `grad/noise_sigma` | `√(noise_sigma2)` | Standard-deviation form of the above. |
| `grad/noise_snr` | `‖ḡ‖ / (σ + eps)` | Signal-to-noise ratio: true gradient magnitude relative to its own noise. Coarse/global, not per-coordinate. |
| `grad/noise_loss_mean`, `grad/noise_loss_var` | mean/variance of the `m` sample losses | Diagnostic on loss variability across the `m` re-sampled batches. Exported but not currently consumed by `fit_scion_constants.py`. |
| `rho/delta_sample_{0,1,2}_fro` | `‖δᵢ‖₂` | Euclidean norm of the deviation `δᵢ = ḡ₋ᵢ − gᵢ`, where `ḡ₋ᵢ` is the mean excluding sample `i`, for each of the first 3 stored raw samples (only 3 kept in full — see note below). |
| `rho/delta_sample_{0,1,2}_star` | `‖δᵢ‖*` | Dual norm of the same deviation vectors. |
| `rho/noise_rho_sample_{0,1,2}` | `‖δᵢ‖* / ‖δᵢ‖₂` | Individual per-sample ρ estimates. |
| `rho/averaged_rho_over_samples` | mean of the 3 `noise_rho_sample_i` | Simple average of the 3 individual ratios. |
| `rho/rho_over_averaged_norms` | `(Σδ*) / (Σδ_fro)` | **This is the ρ estimate `fit_scion_constants.py` uses by default** — ratio of summed numerator/denominator rather than mean-of-ratios; more stable when an individual `‖δᵢ‖₂` is small. |
| `rho/reference_samples` | `m − 1` | Number of independent batch gradients in each leave-one-out reference mean. |
| `rho/noise_step` | `self.n_steps` | Step counter at the time this batch of noise stats was reported. |

**Note on the 3-sample cap:** only the first 3 of the `m` raw gradient samples are kept in
full (`keep_reference = state["m"] < 3`) to compute the δ/ρ comparison, to avoid storing `m`
full-precision gradient copies in memory. For each stored sample, the "true gradient"
stand-in is the mean of the other `m − 1` samples. This leave-one-out construction avoids
shrinking the deviation by including `gᵢ` in its own reference. This is also why
`trace_noise_min_samples` defaults to 3.

## Mapping to the BST paper's constants

| Constant | Source metric | Affected by the pre-2026-06-08 multi-node sync bug? |
|---|---|---|
| **L** | `stats/local_smooth_spec` | No — per-step trace, always correctly DDP-synced. |
| **μ** | Huber-regression slope of `stats/grad_norm_nuc_power_1` vs. `trainer/loss` | No — same reason. |
| **ρ** | `rho/rho_over_averaged_norms` | **Yes** — noise trace; only trust post-fix (or single-node) runs. |
| **σ*** | derived from `grad/noise_sigma2` | **Yes** — same caveat. |

See `scripts/analysis/fit_scion_constants.py` for the fitting pipeline, and
`scripts/analysis/export_scion_trace_wandb.py` for pulling these columns out of W&B.
