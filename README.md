# Alibaba workload pattern forecasting code

This repository contains Python code for an Alibaba Cluster Trace v2018
workload-pattern forecasting workflow. It includes a small core implementation,
an Alibaba topology-only forecasting pipeline, and scripts for direct-horizon,
ablation, and sensitivity experiments. The repository contains source code and
configuration only. Downloaded trace files, derived datasets, model checkpoints,
results, figures, and draft documents are not included.

## Scope

- `src/workload_fmm/`: reusable preprocessing, shape-state, fuzzy-relation,
  transition, BackOff, and graph modules.
- `experiments/alibaba_ours_lh_recursive/`: topology-only recursive forecasting.
- `experiments/alibaba_ours_duibi1_LH/`: direct multi-step forecasting.
- `experiments/alibaba_ours_ablation_recursive/`: five direct multi-step variants.
- `experiments/alibaba_ours_minganxing_KPRλτ/`: K/P/R/decay/time-bin sensitivity.

The Alibaba trace supplies CPU, memory, and total disk-I/O utilization. The two
disk channels in the prepared input are deterministic derivatives of total
disk-I/O utilization. GPU utilization and measured link bandwidth, delay, loss,
or availability are not present. The Alibaba spatial experiments therefore use
observed workload-to-node placement as an **unweighted topology**. Their output
should be described as *topology-only*, not as measured link-quality results.

## Environment and quick check

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python examples/run_demo.py
python -m unittest discover -s tests
```

For a small synthetic check of the main experimental pipeline:

```bash
cd experiments/alibaba_ours_lh_recursive
python run_experiment.py smoke-test --config configs/experiment.json
```

Full training requires substantially more time and storage than the quick check.
Use a PyTorch build compatible with your GPU if running the CUDA configurations.

## Obtain and prepare the data

Download the source trace from the [Alibaba Cluster Trace v2018 project](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018).
From the repository root, the included download and extraction scripts expect
`container_meta.tar.gz` and `container_usage.tar.gz` in
`data/alibaba_v2018/raw/`:

```bash
python scripts/download_alibaba_v2018.py --subset required
python scripts/extract_alibaba_v2018.py --name container_usage
```

The download is large; the extracted CSV needs substantial additional space.
If you obtained the trace separately, place those two archives in the same
`data/alibaba_v2018/raw/` directory before extraction. See
[`data/README.md`](data/README.md) for the expected paths and data protocol.

Then, from `experiments/alibaba_ours_lh_recursive/`, run:

```bash
python run_experiment.py preflight --config configs/experiment.json
python run_experiment.py prepare --config configs/experiment.json --resume
```

The prepare step creates a local portable `dataset.npz` and cohort files under
`inputs/alibaba_selected_200/`. They are ignored by Git. To make that prepared
input available to the other experiments, return to the repository root and run:

```bash
python scripts/prepare_experiment_inputs.py
```

## Experiment entry points

Run each command from the experiment directory shown in the first column.
Configurations are kept with the corresponding experiment.

| Directory | Check command | Full run command |
| --- | --- | --- |
| `alibaba_ours_lh_recursive` | `python run_experiment.py preflight --config configs/portable.json` | `python run_experiment.py run-all --config configs/portable.json --resume` |
| `alibaba_ours_duibi1_LH` | `python run_experiment.py preflight --config configs/portable.json` | `python run_experiment.py run-all --config configs/portable.json --resume` |
| `alibaba_ours_ablation_recursive` | `python run_experiment.py design-check --config configs/ablation.json` | See `python run_experiment.py --help` for variant and device arguments. |
| `alibaba_ours_minganxing_KPRλτ` | `python run_experiment.py preflight --config configs/sensitivity.json` | See `python run_experiment.py --help` for study and device arguments. |

The formal ablation and sensitivity configurations target a CUDA workstation;
their synthetic checks and unit tests can be run separately. The selected
200-workload cohort has sparse colocations, so it cannot identify effects that
require hyperedges of size three or indirect two-hop workload pairs. Treat
those structural ablations as exploratory.

## Tests

Run each suite from its directory:

```bash
python -m unittest discover -s tests
```

The command applies to the repository root and to each experiment directory
containing `tests/`. Unit tests and synthetic checks do not establish that a
full training run reproduces any particular published metric.

## Repository contents

No manuscript, tables, figures, downloaded Alibaba trace, derived dataset,
checkpoint, or result archive is distributed here. All output directories are
excluded by `.gitignore`. Review the `git status` file list before publishing
or adding new files.

## Contributors

- [1JSK1](https://github.com/1JSK1) — Code contributor.
