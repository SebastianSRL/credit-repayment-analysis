from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    cfg = dict(one_hot=list(), binary=list())
    for col in working_train_df.select_dtypes(include="object").columns:
        if working_train_df[col].nunique() == 2:
            cfg["binary"].append(col)
        else:
            cfg["one_hot"].append(col)

    # OrdinalEncoder
    ord_enc = OrdinalEncoder()
    working_train_df[cfg["binary"]] = ord_enc.fit_transform(
        working_train_df[cfg["binary"]]
    )
    working_val_df[cfg["binary"]] = ord_enc.transform(working_val_df[cfg["binary"]])
    working_test_df[cfg["binary"]] = ord_enc.transform(working_test_df[cfg["binary"]])

    # OneHotEncoder
    one_hot_enc = OneHotEncoder(sparse_output=False)
    temp_train_df = one_hot_enc.fit_transform(working_train_df[cfg["one_hot"]])
    temp_val_df = one_hot_enc.transform(working_val_df[cfg["one_hot"]])
    temp_test_df = one_hot_enc.transform(working_test_df[cfg["one_hot"]])

    working_train_df = np.concatenate(
        [working_train_df.drop(columns=cfg["one_hot"]).to_numpy(), temp_train_df],
        axis=1,
    )
    working_val_df = np.concatenate(
        [working_val_df.drop(columns=cfg["one_hot"]).to_numpy(), temp_val_df], axis=1
    )
    working_test_df = np.concatenate(
        [working_test_df.drop(columns=cfg["one_hot"]).to_numpy(), temp_test_df], axis=1
    )

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.
    imputer = SimpleImputer(strategy="median")
    working_train_df = imputer.fit_transform(working_train_df)
    working_val_df = imputer.transform(working_val_df)
    working_test_df = imputer.transform(working_test_df)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    mm_scaler = MinMaxScaler()
    working_train_df = mm_scaler.fit_transform(working_train_df)
    working_val_df = mm_scaler.transform(working_val_df)
    working_test_df = mm_scaler.transform(working_test_df)

    return (
        working_train_df,
        working_val_df,
        working_test_df,
    )
