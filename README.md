# Reactive-power control and capacity-planning benchmark

This repository provides code, trained policies, simulation inputs, and frozen evaluation outputs for risk-aware Volt-VAR control and reactive-power capacity planning on a modified IEEE 33-bus distribution feeder.

The implemented workflow has three linked components:

1. capacity-conditioned worst-group CVaR PPO (WG-CVaR-PPO);
2. an AC power-flow safety projection applied to policy actions; and
3. a three-stage empirical chance-constrained scan of PV-inverter, SVC, and capacitor capacity.

## Included results

- Locked uniform and stress evaluations contain 1,000 shared jobs per set for three independently trained policy seeds.
- The capacity-planning outputs cover 2,000 screened combinations, 300 refined combinations, and 37 confirmed configurations.
- Raw and safety-projected outputs are retained for WG-CVaR and Concat PPO.
- The confirmed minimum-resource configuration is `[0.7, 0.7, 0.0]`, expressed as PV-inverter capacity scale, SVC capacity scale, and capacitor capacity in MVar.

These are simulation results for the included modified IEEE 33-bus system. The resource index is a normalised study metric rather than a monetary cost or life-cycle measure.

## Repository map

| Path | Contents |
|---|---|
| `src/` | Training, evaluation, safety-projection, and capacity-planning code |
| `models/` | WG-CVaR, Concat, and Blind PPO checkpoints for seeds 42, 43, and 44 |
| `data/inputs/` | Network case and load/PV simulation profiles |
| `data/results/locked_controller_evaluation/` | Frozen uniform/stress jobs and controller outputs |
| `data/results/capacity_planning/` | Screening, refinement, confirmation, ablation, and capacity-path outputs |
| `data/derived_metrics/` | Machine-readable summaries derived from the frozen results |
| `scripts/` | Lightweight integrity checks and an execution smoke test |
| `docs/` | Reproducibility protocol and metric definitions |

## Installation

Python 3.10 was used for the frozen evaluation. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Fast verification

Check the model and result structure and recompute the principal aggregate metrics without rerunning power flow:

```bash
python scripts/verify_release.py
```

Run one deterministic environment step with a WG-CVaR checkpoint:

```bash
python scripts/smoke_test.py
```

## Reproduce evaluations

Re-evaluate the frozen uniform and stress job tables:

```bash
python src/run_locked_projection_confirmation.py
```

Rerun the three-stage capacity-planning protocol:

```bash
python src/run_final_capacity_planning.py
```

The complete planning run is computationally intensive. New outputs are written to `reproduced/`; the frozen files under `data/results/` are not overwritten by default. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the evaluation hierarchy and metric conventions.

## Action and capacity definitions

The shared policy receives the network state and capacity vector `theta = [s_pv, s_svc, q_cap]`. It outputs four normalised continuous actions: reactive-power commands for PV inverters at zero-based buses 17, 21, and 24, and for the centralised SVC at zero-based bus 32. The environment maps these commands to device-specific reactive-power limits before AC power flow.

## Licence and provenance

Code is provided under the repository licence. Benchmark and environment provenance is documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). PyTorch checkpoints should only be loaded from trusted sources.
