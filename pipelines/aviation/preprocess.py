"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "Investigation.Type",
    "Country",
    "Injury.Severity",
    "Amateur.Built",
    "Number.of.Engines",
    "Total.Fatal.Injuries",
    "Total.Serious.Injuries",
    "Total.Minor.Injuries",
    "Total.Uninjured",
]
label_column = "Aircraft.damage"

feature_columns_dtype = {
    "Investigation.Type": np.float64,
    "Country": np.float64,
    "Injury.Severity": np.float64,
    "Amateur.Built": np.float64,
    "Number.of.Engines": np.float64,
    "Total.Fatal.Injuries": np.float64,
    "Total.Serious.Injuries": np.float64,
    "Total.Minor.Injuries": np.float64,
    "Total.Uninjured": np.float64,
}
label_column_dtype = {"Aircraft.damage": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print("Input data:", input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/aviation-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    
    inflow_fn = f"{base_dir}/data/aviation-inflow.csv"
    inflow_key = "/".join([input_data.split("/")[3], "aviation_inflow.csv"])
    s3.Bucket(bucket).download_file(inflow_key, inflow_fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=1,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    
    inflow_df = pd.read_csv(
        inflow_fn,
        header=1,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
        )
    
    df = pd.concat([df.tail(len(df) - 1000), inflow_df.head(1000)], ignore_index = True)
    inflow_df = pd.concat([inflow_df.tail(len(inflow_df) - 1000), df.head(1000)], ignore_index = True)
    
    df.to_csv(fn, index=False)
    inflow_df.to_csv(inflow_fn, index=False)
    
    s3.Bucket(bucket).upload_file(fn, key)
    s3.Bucket(bucket).upload_file(inflow_fn, inflow_key)
    
    os.unlink(fn)
    os.unlink(inflow_fn)

#     logger.debug("Defining transformers.")
#     numeric_features = list(feature_columns_names)
#     numeric_features.remove("sex")
#     numeric_transformer = Pipeline(
#         steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
#     )

#     categorical_features = ["sex"]
#     categorical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )

#     preprocess = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features),
#         ]
#     )

    logger.info("Applying transforms.")
    y = df.pop("Aircraft.damage")
    X_pre = df
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

