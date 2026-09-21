# Third-party notices

The MIT license in `LICENSE` applies only to original VMOD code and supporting
material authored for this package. It does not relicense the benchmark network
data, source time-series data, derived time-series arrays, inherited environment
code, or external software dependencies described below. Users are responsible
for complying with the applicable upstream terms.

## Distribution-system benchmark data

`case33_bw.mat`, `case69.mat`, and `case1180zh.mat` are binary MATPOWER-format
representations of published radial distribution test systems. MATPOWER itself
is distributed under a three-clause BSD license, but the MATPOWER documentation
states that case files distributed with MATPOWER are not covered by that BSD
license. The relevant case documentation and primary references are:

- 33-bus case: M. E. Baran and F. F. Wu, "Network reconfiguration in
  distribution systems for loss reduction and load balancing," *IEEE
  Transactions on Power Delivery*, 4(2), 1401-1407, 1989.
  https://doi.org/10.1109/61.25627
- 69-bus case: M. E. Baran and F. F. Wu, "Optimal capacitor placement on
  radial distribution systems," *IEEE Transactions on Power Delivery*, 4(1),
  725-734, 1989. https://doi.org/10.1109/61.19265; see also D. Das, "Optimal
  placement of capacitors in radial distribution system using a Fuzzy-GA
  method," *International Journal of Electrical Power & Energy Systems*,
  30(6-7), 361-367, 2008. https://doi.org/10.1016/j.ijepes.2007.08.004
- 118-bus timing-diagnostic case (`case1180zh.mat`, corresponding to the
  MATPOWER `case118zh` data): D. Zhang, Z. Fu and L. Zhang, "An improved TS
  algorithm for loss-minimum reconfiguration in large-scale distribution
  systems," *Electric Power Systems Research*, 77, 685-694, 2007.
  https://doi.org/10.1016/j.epsr.2006.06.005

Upstream MATPOWER documentation: https://matpower.org/docs/MATPOWER-manual.pdf

## Load and photovoltaic temporal profiles

Files under `data/vmod/profiles33` and `data/vmod/profiles69` are normalized,
feeder-mapped derivatives of the German load and solar series in:

> Open Power System Data. 2020. *Data Package Time series*. Version
> 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06

The package identifies ENTSO-E Transparency as the primary source. OPSD source
page: https://data.open-power-system-data.org/time_series/2020-10-06/

Each released `metadata.json` records the selected columns, frozen time window,
normalization, feeder mapping, source-file SHA-256 digest and derived-array
digests. These system-level temporal shapes are not field measurements from the
33- or 69-bus benchmark feeder. Neither the OPSD source data nor the derived
arrays are relicensed by this package's MIT license.

## Volt-VAR environment lineage

`Env.py` retains its upstream author attribution and has been substantially
modified for the VMOD experiments. The source header identifies the original
preprint, Q. Liu et al., "Reducing learning difficulties: One-step two-critic
deep reinforcement learning for inverter-based Volt-VAR control," arXiv
2203.16289 (2022), https://doi.org/10.48550/arXiv.2203.16289. The subsequent
journal article is Q. Liu et al., "Two-Critic Deep Reinforcement Learning for
Inverter-Based Volt-Var Control in Active Distribution Networks," *IEEE
Transactions on Sustainable Energy*, 15(3), 1768-1781, 2024,
https://doi.org/10.1109/TSTE.2024.3376369.

This attribution does not imply endorsement by the upstream authors and does
not grant rights beyond the applicable upstream terms.

## External software

Python packages installed through `environment.yml` or `requirements.txt` are
not redistributed under this package's MIT license. Each dependency remains
subject to its own license.
