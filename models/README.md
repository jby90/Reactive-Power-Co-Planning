# Released policies

The three `actor.pth` files are PyTorch state dictionaries used in the paper's matched-controller assessment.

| Directory | Observation | Network | Training seed | Frames |
|---|---|---|---:|---:|
| `film/` | 103-state vector plus 3 capacity parameters | Two 256-unit hidden layers with feature-wise linear modulation | 42 | 28,800 |
| `blind/` | 103-state vector | Two 256-unit hidden layers; no capacity input | 42 | 28,800 |
| `concat/` | 103-state vector concatenated with 3 capacity parameters | Two 256-unit hidden layers | 42 | 28,800 |

All policies output four normalised continuous actions: reactive-power commands for PV inverters at zero-based buses 17, 21, and 24, and for the centralised SVC at zero-based bus 32. The environment maps these values to device-specific reactive-power bounds.

The `training/` subdirectories contain `train.csv`, `traintest.csv`, and `trainloss.csv`. These are trajectories from one principal training run, not independent-seed samples. TensorBoard event files and optimiser checkpoints are intentionally excluded because they are unnecessary for evaluation and add approximately 40 MB.

Only load model files from sources you trust. The release scripts load these state dictionaries into explicitly constructed model classes in `src/`.
