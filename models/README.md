# Policy checkpoints

The repository includes three independently trained policies for each family:

| Family | Capacity input | Training objective | Seeds |
|---|---|---|---|
| `wg_cvar` | Concatenated capacity vector | Directional within-group CVaR with worst-group weighting | 42, 43, 44 |
| `concat` | Concatenated capacity vector | Scalar PPO baseline | 42, 43, 44 |
| `blind` | None | Scalar PPO conditioning diagnostic | 42, 43, 44 |

Every seed directory contains `config.json`, `csv/episodes.csv`, `csv/updates.csv`, and `models/checkpoint.pth`. Concat and Blind directories also retain the actor-only state dictionary produced by their trainer. The evaluator loads the complete checkpoint because it records the architecture and capacity bounds required to reconstruct the actor.

All policies produce four normalised reactive-power commands for three PV inverters and one SVC. Only load PyTorch checkpoints from sources you trust.
