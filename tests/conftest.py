import os

import pandas as pd
from pytest import fixture

from src import config
from src.data_utils import get_datasets


@fixture(scope="session", autouse=True)
def app_train() -> pd.DataFrame:
    """Load and return train dataset."""
    if not os.path.exists(config.DATASET_TRAIN):
        app_train_df, _, _ = get_datasets()
    else:
        app_train_df = pd.read_csv(config.DATASET_TRAIN)

    return app_train_df


@fixture(scope="session", autouse=True)
def app_test() -> pd.DataFrame:
    """Load and return test dataset."""
    if not os.path.exists(config.DATASET_TEST):
        _, app_test_df, _ = get_datasets()
    else:
        app_test_df = pd.read_csv(config.DATASET_TEST)

    return app_test_df
