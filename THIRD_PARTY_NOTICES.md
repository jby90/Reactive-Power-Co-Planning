# Third-party notices

## IEEE 33-bus benchmark

`data/inputs/case33_bw.mat` is a MATPOWER-format representation of the radial 33-bus distribution test system associated with:

> Baran, M. E. and Wu, F. F. (1989). Network reconfiguration in distribution systems for loss reduction and load balancing. *IEEE Transactions on Power Delivery*, 4(2), 1401-1407. https://doi.org/10.1109/61.25627

MATPOWER is distributed under a three-clause BSD licence. Users should consult the current MATPOWER distribution for its complete licence and case documentation.

## Volt-VAR environment lineage

`src/Env.py` retains attribution to the environment lineage developed for:

> Liu, Q., Guo, Y., Deng, L., Liu, H., Li, D., Sun, H. and Huang, W. (2024). Two-critic deep reinforcement learning for inverter-based Volt-VAR control in active distribution networks. *IEEE Transactions on Sustainable Energy*, 15(3), 1768-1781. https://doi.org/10.1109/TSTE.2024.3376369

The file has been modified for capacity conditioning, PV inverter capability limits, SVC scaling, shunt-capacitor placement, deterministic scenario handling, and pandapower compatibility. This notice does not imply endorsement by the original authors or grant rights beyond applicable upstream terms.

## Load and PV profile lineage

The released synthetic load and PV inputs inherit the normalised 96-point daily fluctuation template used by the Volt-VAR environment above. That environment attributes the template to:

> Liu, H. and Wu, W. (2021). Two-stage deep reinforcement learning for inverter-based Volt-VAR control in active distribution networks. *IEEE Transactions on Smart Grid*, 12(3), 2037-2047. https://doi.org/10.1109/TSG.2020.3041620

The present release expands the template across feeder loads and PV inverters and includes 370 fixed-seed synthetic daily realisations generated with independent multiplicative factors sampled from `U(0.8, 1.2)`. These arrays are benchmark simulation inputs and must not be described as field measurements.
