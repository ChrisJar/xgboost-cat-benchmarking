# Xgboost Categorical Benchmarking

### Environment Creation
```
mamba create -n xgb-cat-bench -c rapidsai-nightly -c conda-forge python=3.10 cudatoolkit=11.8 rapids-xgboost jupyterlab cudf=23.04 dask-cudf dask-cuda 'ucx-proc=*=gpu' ucx-py
```

### Memory Usage Stats
Using the experimental categorical feature currently uses about 78GB of GPU memory on 3/40th of the Criteo Dataset. So, it should roughly take 14 80GB A100s to train on the full dataset.

TODO: Get an estimate for GPUs needed without using the experimental categorical feature.