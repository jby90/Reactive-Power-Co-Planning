# Variable-capacity model-informed OPF distillation (VMOD)

This package contains the code, frozen temporal inputs, trained policies and
machine-readable evidence for capacity-conditioned raw Volt--VAR control on
33- and 69-bus distribution benchmarks.

## Key design

- Selected teacher voltage margin: `0.003` pu.
- Online execution: one actor forward pass, physical MVAr conversion and device clipping.
- Principal 33-bus evidence stages: 40 fitting days and 10 margin-calibration days from the 366-day development window beginning 1 January 2017, followed by 17 capacity-selection days and 349 confirmation days from a non-overlapping window beginning 2 January 2018.
- Statistical unit: day within each independently trained seed; shared days are not pooled across seeds.
- Capacity result: lowest confirmed tested point on a development-frozen physical path, not a global or economic optimum.
- Direct representation reuse failed on the 69-bus case. The final feeder-specific adaptation standardises the training observations, retains only active-load features, and adds 1,344 AC-OPF examples from 14 tail days exposed by the failed 2018 confirmation.
- The adapted 69-bus P08/P09 pair, five policy hashes and data manifests were frozen before an untouched 366-day window beginning 3 January 2019 was opened. P09 recorded 0/1,830 event seed-days and P08 recorded one. This is feeder-adapted replication, not zero-shot transfer.
- The included 118-bus run is an architecture-and-AC-solver timing diagnostic only, not a control or planning validation.

## Environment

```bash
conda env create -f environment.yml
conda activate vmod
pytest -q
```

## Frozen evidence

- `runs/VMOD_BIDIRECTIONAL_MARGIN_STUDY_20260920/`: margin calibration.
- `runs/VMOD_PATH_MARGIN_CALIBRATION_20260921/`: path-wide margin decision.
- `runs/VMOD_EXTERNAL2018_MARGIN_0p003_20260921/`: 33-bus training, selection, confirmation, baselines and audits.
- `runs/VMOD_ENV69_ADAPTED_FINAL_20260921/`: 69-bus adaptation audit, merged data, final policies and untouched-window fixed-boundary confirmation.
- `runs/VMOD_PROTOCOL_EXTERNAL2019_ENV69_FINAL_20260921/`: frozen 2019 confirmation protocol for the adapted 69-bus P08/P09 pair.
- `runs/VMOD_ENV69_EXTERNAL2018_MARGIN_0p003_20260921/calibration/`: the nine compact development-calibration summaries that triggered the audited 69-bus path extension; no external-selection result is included there.
- `runs/VMOD_FINAL_STUDY_20260921/`: claim gates and figure source data.
- `SHA256SUMS.csv`: hashes for every released file.

To regenerate figures into `outputs/figures`, set `VMOD_FIGURE_OUT` and run:

```bash
set VMOD_FIGURE_OUT=outputs/figures
python make_vmod_figures.py --figures all
```

The package reports empirical simulation evidence only. It does not provide a
formal safety guarantee or an economically calibrated equipment optimum.

Original VMOD code is licensed under `LICENSE`. Benchmark networks, derived
temporal profiles and inherited environment code remain subject to their
upstream terms; see `THIRD_PARTY_NOTICES.md`.
