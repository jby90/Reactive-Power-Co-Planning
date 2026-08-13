# Data dictionary

## Inputs

`inputs/` contains the modified IEEE 33-bus MATPOWER case and the load/PV arrays required by the environment. The network case follows the Baran-Wu benchmark. The load and PV arrays are synthetic simulation inputs, not field measurements: they inherit a normalised 96-point daily fluctuation template from the Volt-VAR environment lineage documented in `THIRD_PARTY_NOTICES.md`, and the bus/device-level arrays contain 370 fixed-seed realisations generated with independent multiplicative factors sampled from `U(0.8, 1.2)`. Repeated base-template blocks retained for environment compatibility are removed by exact joint load/PV hashing when constructing the 200-profile confirmation set. The repository contains no personal or human-participant data.

## Locked controller evaluation

`results/locked_controller_evaluation/` contains the immutable `uniform1000.csv` and `stress1000.csv` job tables. Subdirectories store episode-level results for WG-CVaR and Concat, before and after AC safety projection, for seeds 42, 43, and 44. `locked_summary.csv` and `locked_paired_comparison.csv` provide aggregate summaries.

## Capacity planning

`results/capacity_planning/` contains:

- `grid_candidates.csv`: the full 20 x 20 x 5 capacity grid;
- `screen/`, `refine/`, and `confirm/`: stage-level candidate and seed summaries;
- `final_optimum.csv`: the confirmed minimum-resource configuration;
- `ablation/`: raw/projected WG-CVaR and Concat results at selected configurations;
- `risk_path_raw/`: raw-policy risk along the predefined capacity path;
- `locked_days/`: shared day indices used for screening, refinement, and profile-unique confirmation;
- `delta_*`: additional evaluations used to replace duplicated confirmation profiles without rerunning valid earlier stages.

## Derived metrics

`derived_metrics/` contains compact, machine-readable tables derived from the episode-level outputs. They cover controller comparison, training traces, projection ablation, capacity-path behaviour, configuration composition, resource efficiency, and a representative safety-projection trace.

Some raw result tables retain simulator-oriented column names. Paths recorded by the original orchestration have been normalised to repository-relative paths.
