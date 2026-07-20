# Risk-aware reactive-power co-planning for cleaner PV integration

This repository contains the public code, trained policies, locked evaluation data, planning outputs, and figure source data for:

> **Risk-Aware Reactive-Power Co-Planning with Physics-Guided Safety Projection for Resource-Efficient PV Integration**

The current release implements a deployment-consistent workflow with three linked components:

1. capacity-conditioned worst-group CVaR PPO (WG-CVaR-PPO) for Volt-VAR control;
2. a selectively activated AC power-flow safety projection; and
3. a three-stage empirical chance-constrained scan of inverter, SVC, and capacitor capacity.

The earlier FiLM-based release is retained only for provenance under [`legacy/film-release-v1/`](legacy/film-release-v1/ARCHIVED.md). It is not the method or evidence base used by the current manuscript.

## Headline released results

- On locked uniform and stress sets, projected WG-CVaR reduced pooled mean line loss by **3.74%** and **4.07%**, respectively, relative to projected Concat PPO.
- Both projected controllers recorded **0/3,000 seed-job voltage events** on each locked set.
- The final scan evaluated **2,000** capacity combinations, refined **300**, and confirmed **37** configurations using three independently trained WG-CVaR policies.
- The confirmed minimum-resource point was `[0.7, 0.7, 0.0]`, where the entries are PV-inverter capacity scale, SVC capacity scale, and capacitor capacity in MVar.
- Relative to the predefined high-redundancy case `[1.5, 1.5, 1.0]`, this point reduced the normalised resource index by **53.95%** and daily line-loss energy by **62.88%**.

These are simulation results for the released modified IEEE 33-bus study. The resource index is not a monetary cost or life-cycle footprint.

## Repository map

| Path | Contents |
|---|---|
| `src/` | Training, evaluation, safety-projection, planning, and plotting code |
| `models/` | Three WG-CVaR, Concat, and Blind PPO checkpoints plus training records |
| `data/inputs/` | Network case and load/PV profiles used by the simulator |
| `data/results/locked_controller_evaluation/` | Frozen uniform/stress jobs and controller outputs |
| `data/results/capacity_planning/` | Screening, refinement, confirmation, ablation, and capacity-path outputs |
| `data/figure_source/` | Source tables for all released manuscript figures |
| `figures/` | Submission figures in PNG, PDF, and SVG formats |
| `scripts/` | Lightweight release verification and smoke tests |
| `legacy/film-release-v1/` | Superseded FiLM-era public release |

## Installation

Python 3.10 was used for the locked evaluation. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Fast verification

The verification script checks the released file structure and recomputes the reported percentages from the CSV files without rerunning power flow:

```bash
python scripts/verify_release.py
```

An environment and checkpoint smoke test is available with:

```bash
python scripts/smoke_test.py
```

## Reproduce key analyses

Regenerate the locked controller comparison from the released checkpoints and frozen 1,000-job tables:

```bash
python src/run_locked_projection_confirmation.py
```

Rerun the complete three-stage capacity-planning protocol:

```bash
python src/run_final_capacity_planning.py
```

This full planning run is computationally intensive. It writes new outputs to `reproduced/` and does not overwrite the released evidence in `data/results/`.

Regenerate the result figures from the frozen outputs:

```bash
python src/make_final_cleaner_energy_figures.py
python src/plot_final_training_reward.py
```

Generated figures are written to `figures/generated/`. Further protocol details are provided in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Released action and capacity definitions

The shared policy receives the network state and capacity vector `theta = [s_pv, s_svc, q_cap]`. It outputs four normalised continuous actions: reactive-power commands for PV inverters at zero-based buses 17, 21, and 24, and for the centralised SVC at zero-based bus 32. The environment maps these commands to device-specific reactive-power limits before AC power flow.

## Licence and citation

Code is released under the repository licence. Third-party notices remain in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). If you use this release, please cite the manuscript using [`CITATION.cff`](CITATION.cff); update the citation with the journal DOI once available.
