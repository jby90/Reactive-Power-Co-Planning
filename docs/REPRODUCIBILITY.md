# Reproducibility guide

## Evaluation hierarchy

The repository preserves three distinct evaluation levels.

1. **Locked controller evaluation:** 1,000 shared jobs for each of the uniform and stress sets, evaluated by policy seeds 42, 43, and 44. Raw and AC-projected WG-CVaR and Concat outputs are retained.
2. **Three-stage capacity planning:** 2,000 candidates are screened, 300 are refined, and 37 are confirmed. A candidate is feasible only when every WG-CVaR seed satisfies the empirical event-day threshold.
3. **Profile-unique confirmation:** each confirmed configuration is evaluated on the same 200 distinct held-out daily profiles for every policy seed.

The controller comparison and capacity-boundary result answer different questions. The former compares operating loss under matched projected safety. The latter uses WG-CVaR's own worst-seed feasibility rule and does not depend on WG-CVaR outperforming Concat for every training seed.

## Risk and energy conventions

- A voltage-event job contains at least one 15-minute step outside `[0.95, 1.05]` pu or a power-flow failure.
- Planning uses an empirical event-day tolerance of `epsilon = 0.01` for every policy seed.
- The AC projection uses the internal guard band `[0.9505, 1.0495]` pu.
- Daily line-loss energy is the sum of simulated active line loss over 96 15-minute steps.
- Projection intervention is the fraction of control steps whose raw policy action is modified.
- The resource index is `50*s_pv + 50*s_svc + 2*q_cap`; it is a transparent normalised study index, not CAPEX.

## Checkpoints

Each `models/<family>/seed<seed>/` directory contains the exact configuration, training CSV files, and checkpoint used in the frozen evaluation. PyTorch checkpoint files should only be loaded from trusted sources.

## Expected compute

`scripts/verify_release.py` completes in seconds and requires no GPU. Replaying the locked jobs requires repeated AC power flow and benefits mainly from CPU parallelism. Training and the complete 2,000-candidate planning pipeline are substantially more expensive; GPU acceleration is used for policy training and inference, while AC power-flow evaluation remains CPU-bound.

## Output isolation

Reproduction commands write to `reproduced/`. The frozen data under `data/results/` are never overwritten by default.
