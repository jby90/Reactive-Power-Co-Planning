# Reactive-Power Co-Planning

This repository contains the code, frozen inputs, trained policies, job tables, and result tables for capacity-conditioned reactive-power control and resource screening on IEEE distribution-network benchmarks.

The released evidence separates three questions:

1. whether a learned controller remains safe at a specified device-capacity vector;
2. how an AC power-flow safety projection changes controller actions and outcomes; and
3. which capacity vectors pass the fixed multi-seed screening and confirmation protocol.

The principal 33-bus study uses 366 public 15-minute German load and solar temporal profiles from the Open Power System Data time-series package (2020-10-06). These system-level temporal shapes are mapped to the fixed spatial allocations of the benchmark feeder; they are not field measurements from that feeder. The frozen split contains 50 training, 17 selection, and 299 confirmation days.

## Repository layout

- `src/`: controller training, evaluation, safety projection, local AC-OPF comparison, mismatch analysis, and summary utilities.
- `data/inputs/`: benchmark cases, processed temporal profiles, metadata, and frozen day splits.
- `data/jobs/`: capacity-grid and confirmation job tables.
- `data/results/`: source result tables used for the reported analyses.
- `models/`: final multi-seed policies, ablation policies, sensitivity policies, portability policies, and policy-initialisation weights.
- `tests/`: focused tests for profile loading, day-table loading, safety projection, and local AC-OPF.
- `data/release_manifest.json`: SHA-256 and byte size for every released file except the manifest itself.

## Environment

Python 3.11 is recommended. Install the tested package versions with:

```bash
python -m pip install -r requirements.txt
```

PyTorch wheels are platform-specific. The recorded experiments used PyTorch 2.10.0 with CUDA 12.6; CPU evaluation is also supported.

## Verify the release

From the repository root:

```bash
python scripts/verify_release.py
python -m pytest -q
```

The verifier checks every manifest hash, split disjointness, array dimensions, and the presence of all principal five-seed checkpoints.

## Reproduce a held-out evaluation

This command evaluates the seed-42 worst-group CVaR policy on the 299 confirmation days at the first all-method, all-seed passing vector C105, `[0.45, 0.5625, 0]`:

```bash
python src/evaluate_crdc_policy.py \
  --run_dir models/WG_CVAR_PPO/env33_seed42_gamma0.9/20260918-confirm-opsd-wg-ref-seed42 \
  --theta 0.45,0.5625,0 \
  --days_metadata data/inputs/confirmation_days.csv \
  --device cpu \
  --out_dir reproduced/wg_seed42_c105
```

The command writes `daily.csv` and `summary.json`. Additional commands for the capacity grid, projection, local AC-OPF, model mismatch, and capacitor mechanism are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Key frozen evidence

- C100 `[0.45, 0.375, 0]` is rejected by the confirmation protocol.
- C105 `[0.45, 0.5625, 0]` is the first vector on the frozen path with zero event days for both controllers, before and after projection, across five seeds and 299 days per seed.
- A zero count in one 299-day seed corresponds to a one-sided exact 95% upper event-rate bound of 0.9969%, not a proof of universal safety.
- The local AC-OPF comparator is a feasible nonconvex operational reference, not a certified global lower bound.
- The 69-bus results test protocol portability and are not used to claim a new capacity boundary.

## License and data attribution

Code is released under the license in [LICENSE](LICENSE). Third-party software and data notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The OPSD source URL, DOI, transformation, and SHA-256 checksums are recorded in `data/inputs/metadata.json`.
