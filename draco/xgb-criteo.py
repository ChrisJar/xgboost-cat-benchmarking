import argparse
import sys
import time

import dask_cudf
import xgboost as xgb

from dask.distributed import Client

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--scheduler-file")
    return vars(parser.parse_args())


def attach_to_cluster(scheduler_file):
    client=None
    try:
        with open(scheduler_file) as fp:
            print(fp.read())
        client = Client(scheduler_file=scheduler_file)
        print('Connected!', flush=True)
    except OSError as e:
        sys.exit(f"Unable to create a Dask Client connection: {e}")

    return client


def run_bench(config):
    client = attach_to_cluster(config["scheduler_file"])

    X_train = dask_cudf.read_parquet(
        config["data_dir"],
        split_row_groups=True
    )

    y_train = X_train["label"]
    X_train = X_train.drop("label", axis=1)

    print("Creating DMatrix", flush=True)

    dtrain = xgb.dask.DaskQuantileDMatrix(client, X_train, y_train, enable_categorical=True)
    del X_train
    del y_train

    params = {
        "tree_method": "gpu_hist",
        "objective": "binary:logistic",
        "max_cat_to_onehot": 1
    }

    print("Beggining Training", flush=True)

    t0 = time.time()
    output = xgb.dask.train(
            client,
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train")],
    )
    t1 = time.time()
    training_time = t1-t0
    print("Training Time: {}".format(training_time))


if __name__ == "__main__":
    config = parse_args()
    run_bench(config)