# Reactive-Power Co-Planning for Cleaner PV Integration

Research code, processed benchmark data, trained policies, and source results for **"Resource-Efficient Reactive-Power Co-Planning for Cleaner PV Integration in Active Distribution Networks"**.

The repository implements a bi-level workflow:

1. a capacity-conditioned FiLM-PPO controller coordinates three PV smart inverters and one centralised SVC on a modified IEEE 33-bus feeder;
2. an outer Monte Carlo scan evaluates inverter, SVC, and shunt-capacitor capacities under empirical day-level chance constraints; and
3. the released result tables support the paper's controller, planning, resource-efficiency, and PV-feasibility analyses.

## Repository contents

| Path | Contents |
|---|---|
| `src/Env.py` | AC power-flow environment and reactive-power device models |
| `src/PPO_theta_robust_film_curriculum.py` | FiLM-PPO training with curriculum domain randomisation |
| `src/PPO_theta_robust.py` | Blind and concatenation-conditioned PPO baselines |
| `src/chance_constraint_capacity_planning_film.py` | Chance-constrained outer capacity scan |
| `src/cleaner_energy_assessment.py` | Matched-day controller and resource-efficiency assessment |
| `data/inputs/` | Modified 33-bus case and normalised load/PV time series |
| `data/results/` | Candidate scans, selected plans, figure source data, and run metadata |
| `models/` | Released FiLM, Blind, and Concat policy weights plus training logs |
| `figures/manuscript/` | Figures used in the submitted manuscript |
| `scripts/` | Evaluation, figure generation, smoke testing, and release verification |

See [`data/README.md`](data/README.md) and [`models/README.md`](models/README.md) before interpreting the files.

## Installation

Python 3.10 or 3.11 is recommended. The release was verified with Python 3.11, PyTorch 2.9.1, and pandapower 3.1.2.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick verification

Run one IEEE 33-bus power-flow step, load all policy weights, and verify the published key values:

```bash
python scripts/verify_release.py
```

Regenerate the cleaner-energy evidence figures from the released CSV files:

```bash
python scripts/make_cleaner_energy_paper_figures.py
```

Generated PNG, PDF, and SVG files are written to `figures/generated/`.

## Reproduce the fixed-hardware comparison

The following command evaluates FiLM-PPO, Blind PPO, and Concat PPO with the same hardware, PV profile, and 30 held-out days. Results are placed in a timestamped directory under `reproduced/controller_comparison/`.

```bash
python src/cleaner_energy_assessment.py \
  --actor_run_dir models/film \
  --blind_run_dir models/blind \
  --concat_run_dir models/concat \
  --pv_generation_levels 1.0 \
  --pv_s_levels 0.7 \
  --svc_levels 1.1 \
  --cap_levels 0.0 \
  --cap_buses 20,8 \
  --n_days_coarse 5 \
  --n_days_final 30 \
  --refine_topk 1 \
  --eps_v 0.05 \
  --eps_pf 0.0 \
  --cost_pv 50 \
  --cost_svc 50 \
  --cost_cap 2 \
  --seed 20260714 \
  --out_dir reproduced/controller_comparison
```

The published comparison is in `data/results/cleaner_energy/controller_comparison/best_plans.csv`. Under this configuration, the annualised line-loss estimates are 383.0 MWh for FiLM-PPO, 424.4 MWh for Blind PPO, and 379.0 MWh for Concat PPO; none of the three policies produced a day-level voltage event in the 30 sampled days.

## Reproduce the outer capacity scan

This command evaluates the 20 x 20 x 5 capacity mesh used in the released planning table. It is substantially slower than the quick verification.

```bash
python src/chance_constraint_capacity_planning_film.py \
  --actor_run_dir models/film \
  --env 33 \
  --seed 42 \
  --res 20 \
  --pv_min 0.7 --pv_max 1.5 \
  --svc_min 0.7 --svc_max 1.5 \
  --cap_levels 0,0.25,0.5,0.75,1.0 \
  --cap_buses 20,8 \
  --n_scenarios_coarse 5 \
  --n_scenarios 30 \
  --topk_refine 150 \
  --stride 2 \
  --eps_v 0.01 --eps_pf 0.01 \
  --cost_pv 50 --cost_svc 50 --cost_cap 2 \
  --out_dir reproduced/capacity_scan
```

The archived 2,000-candidate output is `data/results/capacity_planning/summary_candidates.csv`.

## Interpretation boundaries

- The principal controller training comparison uses one training seed (`42`). Rolling bands in the training figures are not confidence intervals across independently trained policies.
- Empirical risk is the fraction of sampled days containing at least one voltage event or power-flow failure. Thirty zero-event days do not prove zero population risk.
- The asset-cost index uses benchmark weights (`50`, `50`, and `2`) and is not market CAPEX, currency, lifecycle impact, or an emissions estimate.
- Most controller comparisons use a simulator-derived line-loss metric. Annualised MWh values multiply the matched-day mean by 365 and are not utility-metered annual losses.
- The study covers one modified IEEE 33-bus feeder and a bounded family of capacity and PV scenarios. It does not establish universal hosting capacity or deployment performance.

## Citation

If you use this release, please cite the associated article after publication. Citation metadata are also available in [`CITATION.cff`](CITATION.cff).

## Licence and third-party material

Original repository code is released under the MIT License; see [`LICENSE`](LICENSE). Benchmark or adapted material remains subject to its original terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the IEEE 33-bus case and the environment attribution. The processed data are supplied for transparent verification; consult [`data/README.md`](data/README.md) before redistribution.

## Contact

Questions about the release can be directed to the corresponding author, Boyin Jin (`boyin_jin@just.edu.cn`).
