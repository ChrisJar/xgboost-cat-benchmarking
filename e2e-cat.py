import cudf
import dask_cudf
import xgboost as xgb
import dask.dataframe as dd

import time

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from sklearn.metrics import accuracy_score, log_loss

train_data_dir = "/datasets/criteo/crit_orig_pq_10days"
test_data_dir = "/raid/cjarrett/data/criteo/crit_orig_pq_1day"

cluster = LocalCUDACluster(
    n_workers=1,
    CUDA_VISIBLE_DEVICES="2",
    protocol="ucx",
    jit_unspill=True,
    rmm_pool_size="70GiB",
    device_memory_limit="78GB",
)
client = Client(cluster)

train_ddf = dask_cudf.read_parquet(train_data_dir, split_row_groups=True)

def transform_df(ddf):
    ddf = ddf.fillna(0)

    for n in range(1,27):
        col = "C" + str(n)
        ddf[col] = ddf[col].astype("category")
    return ddf

train_ddf = transform_df(train_ddf)
X_train, y_train = train_ddf.drop("label", axis=1), train_ddf["label"]
del train_ddf

dtrain = xgb.dask.DaskQuantileDMatrix(client, X_train, y_train, enable_categorical=True)
del X_train
del y_train

params = {
    "tree_method": "gpu_hist",
    "objective": "binary:logistic",
    "max_cat_to_onehot": 1
}

t0 = time.time()
output = xgb.dask.train(
        client,
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train")],
)
t1 = time.time()
print("Training Time: {}".format(t1-t0))

del dtrain

test_ddf = dd.read_parquet(test_data_dir, split_row_groups=True)

test_ddf = transform_df(test_ddf)
X_test, y_test = test_ddf.drop("label", axis=1), test_ddf["label"]
del test_ddf

y_test_pred_prob = xgb.dask.inplace_predict(client, output, X_test)
y_test_pred_val = y_test_pred_prob>=0.5

test_acc = accuracy_score(y_test.compute(), y_test_pred_val.compute())
test_log_loss = log_loss(y_test.compute(), y_test_pred_prob.compute())

print("Test Accuracy: {}".format(test_acc))
print("Test Log Loss: {}".format(test_log_loss))
