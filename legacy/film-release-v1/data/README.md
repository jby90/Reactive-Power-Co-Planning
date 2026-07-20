# Data documentation

## Input files

| File | Shape | Meaning |
|---|---:|---|
| `inputs/case33_bw.mat` | MATPOWER case | Modified IEEE 33-bus radial distribution feeder |
| `inputs/load96.npy` | `(38496,)` | Normalised scalar load profile at 15-minute resolution |
| `inputs/gen96.npy` | `(38496,)` | Normalised scalar PV profile at 15-minute resolution |
| `inputs/two33load.npy` | `(38496, 32)` | Per-load multipliers used by the simulator |
| `inputs/two33gen.npy` | `(38496, 3)` | Per-inverter PV multipliers used by the simulator |

The arrays contain 401 days of 96 quarter-hour samples. `two33load.npy` and `two33gen.npy` are deterministic processed arrays. They are reproduced by broadcasting `load96.npy` and `gen96.npy` to the 32 loads and three PV inverters, then applying NumPy `default_rng(0)` uniform multipliers in `[0.8, 1.2]` to the first 370 days. The implementation is retained in `src/Env.py`.

The current project archive does not retain a separate upstream provenance record for the one-dimensional normalised profiles. They are therefore released as the exact research inputs used in the reported simulations, not represented as raw measurements from a named utility or location.

## Result files

`results/capacity_planning/summary_candidates.csv` contains all 2,000 combinations in the 20 x 20 x 5 planning mesh. The principal fields are:

| Field | Definition |
|---|---|
| `pv_s_scale` | PV inverter apparent-power capacity multiplier |
| `svc_q_scale` | Centralised SVC reactive-power capacity multiplier |
| `cap_total_mvar` | Total shunt-capacitor capacity allocated to the candidate buses |
| `pr_violation_any` | Fraction of sampled days with at least one voltage-band violation |
| `pr_pf_fail_any` | Fraction of sampled days with at least one power-flow failure |
| `loss_mean_pos` | Mean positive simulator line-loss proxy over the evaluated samples |
| `capex` | Normalised asset-cost index, not currency or market CAPEX |
| `pass` | Coarse or refined Monte Carlo evaluation stage |
| `feasible` | Indicator under the specified empirical chance limits |

`results/cleaner_energy/` contains the matched-day assessments used in the paper:

- `resource_efficient/`: selected low-redundancy configuration;
- `high_redundancy/`: deliberately high-capacity comparison;
- `pv_envelope/`: evaluated PV-generation boundary;
- `controller_comparison/`: FiLM, Blind, and Concat PPO at fixed hardware.

Each directory includes `candidate_results.csv`, `best_plans.csv`, and `metadata.json`. The metadata records the random seed, sampled day indices, candidate grid, risk definition, cost weights, and annualisation convention.

## Checks and limitations

- All values are simulation outputs; there are no human participants or personal data.
- Day indices are sampled without replacement using seed `20260714`.
- `annual_loss_mwh = daily_loss_mwh x 365`; this annualisation does not model seasonal reweighting beyond the sampled-day average.
- `daily_pv_energy_mwh` is accepted simulated PV injection. The environment has no active-power curtailment action.
- The data do not quantify embodied carbon, lifecycle footprints, market prices, equipment ageing, communication delay, switching cost, or protection constraints.

## Reuse

These processed files are provided with the repository for verification of the associated study. `case33_bw.mat` is based on the standard Baran-Wu/MATPOWER test case and remains subject to applicable upstream terms. The repository's MIT licence applies to original code, not automatically to third-party benchmark material.
