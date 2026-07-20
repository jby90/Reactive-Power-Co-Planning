# Data dictionary

## Inputs

`inputs/` contains the modified IEEE 33-bus Matpower case and the load/PV arrays required by the environment. Values are simulation inputs; the release contains no personal or human-participant data.

## Locked controller evaluation

`results/locked_controller_evaluation/` contains the immutable `uniform1000.csv` and `stress1000.csv` job tables. Subdirectories store episode-level results for WG-CVaR and Concat, before and after AC safety projection, for seeds 42, 43, and 44. `locked_summary.csv` and `locked_paired_comparison.csv` are the manuscript-level summaries.

## Capacity planning

`results/capacity_planning/` contains:

- `grid_candidates.csv`: the full 20 x 20 x 5 capacity grid;
- `screen/`, `refine/`, and `confirm/`: stage-level candidate and seed summaries;
- `final_optimum.csv`: the confirmed minimum-resource configuration;
- `ablation/`: raw/projected WG-CVaR and Concat results at selected configurations;
- `risk_path_raw/`: raw-policy risk along the predefined capacity path;
- `locked_days/`: shared day indices used for screen, refine, and profile-unique confirmation;
- `delta_*`: additional evaluations used to repair duplicated confirmation profiles without rerunning valid earlier stages.

## Figure source data

`figure_source/` contains one CSV table per released manuscript figure. These tables are derived from the episode-level files and are provided to make plotting claims directly auditable.

Some raw result tables retain simulator-oriented column names. Paths recorded by the original orchestration have been normalised to repository-relative paths for public release.
