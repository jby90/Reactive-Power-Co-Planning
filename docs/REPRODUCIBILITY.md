# Reproducibility guide

Run all commands from the repository root. Paths in released checkpoint configurations are repository-relative; the evaluator also resolves profile arrays by filename under `data/inputs`.

## 1. Integrity and tests

```bash
python scripts/verify_release.py
python -m pytest -q
```

## 2. Frozen data split

The split is defined by:

- `data/inputs/training_days.csv`: 50 days;
- `data/inputs/selection_days.csv`: 17 days;
- `data/inputs/confirmation_days.csv`: 299 days.

All rows refer to the same processed 366-day OPSD profile arrays. Source attribution, normalization, mapping, dimensions, and checksums are in `data/inputs/metadata.json`.

## 3. Controller evaluation

Example for the seed-42 worst-group CVaR policy at C105:

```bash
python src/evaluate_crdc_policy.py \
  --run_dir models/WG_CVAR_PPO/env33_seed42_gamma0.9/20260918-confirm-opsd-wg-ref-seed42 \
  --theta 0.45,0.5625,0 \
  --days_metadata data/inputs/confirmation_days.csv \
  --device cpu \
  --out_dir reproduced/wg_seed42_c105
```

Replace the run directory with the corresponding `CONCAT_SCALAR_CORRECTED` directory to evaluate the comparison controller. The evaluator records daily event indicators, voltage extrema, power-flow failures, and line-loss energy.

## 4. Capacity and projection analyses

The exact candidate and execution tables are in `data/jobs`. Batch evaluation is provided by:

```bash
python src/evaluate_capacity_job_table.py --help
```

The AC power-flow safety projection implementation is `src/physics_safety_projection.py`. Frozen projection results, activation counts, failure counts, and runtime tables are under `data/results/opsd_secondary_evaluations`.

## 5. Local AC-OPF comparison

The nonconvex local comparator is implemented in `src/ac_opf_vvc_baseline.py`. Both profile arrays must be supplied explicitly:

```bash
python src/ac_opf_vvc_baseline.py --help
```

The final representative-vector and learned-controller results are under:

- `data/results/opsd_secondary_evaluations/ac_opf_final_representatives`;
- `data/results/opsd_secondary_evaluations/ac_opf_final_learned_controllers`;
- `data/results/opsd_secondary_evaluations/ac_opf_final_comparison`.

## 6. Model mismatch and mechanism analyses

Use the following entry points for new runs:

```bash
python src/evaluate_projection_model_mismatch.py --help
python src/evaluate_capacitor_mechanism.py --help
```

The fixed mismatch definitions and frozen outputs are in `data/results/opsd_secondary_evaluations/model_mismatch_final`. Capacitor-pair jobs and outputs are in the corresponding `capacitor_mechanism_jobs` and `capacitor_mechanism` directories.

## 7. Evidence provenance

`data/results/final_evidence_manifest.csv` and `.json` identify the frozen analysis evidence from which this release was assembled. `data/release_manifest.json` independently verifies the public package. Environment metadata is under `data/results/environment`.
