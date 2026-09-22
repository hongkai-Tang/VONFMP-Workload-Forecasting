# Data inputs

This repository does not redistribute Alibaba trace records or derived data.
Obtain Alibaba Cluster Trace v2018 from the [official project](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018).

Expected local files after download and extraction:

```text
data/alibaba_v2018/raw/container_meta.tar.gz
data/alibaba_v2018/raw/container_usage.tar.gz
data/alibaba_v2018/raw/container_usage.csv
```

`experiments/alibaba_ours_lh_recursive/configs/experiment.json` selects a
seeded metadata candidate pool and prepares up to 200 workloads on a 60-second
axis. The main experiment's `prepare` command creates the portable local cache
and cohort description. The other experiment configurations use that fixed
cohort. Raw and derived inputs stay local and are ignored by Git.

The trace does not provide separate disk read/write, GPU utilization, or
measured network-link quality fields for this workflow. Prepared `disk_i` and
`disk_o` are derived from the total disk-I/O percentage, and spatial inputs
use deployment topology only.
