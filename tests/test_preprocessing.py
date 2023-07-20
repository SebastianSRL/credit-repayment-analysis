import numpy as np
import pandas as pd

from src.data_utils import get_feature_target, get_train_val_sets
from src.preprocessing import preprocess_data


def test_preprocess_data(app_train: pd.DataFrame, app_test: pd.DataFrame):
    """Test the preprocess_data function."""
    X_train, y_train, X_test, _ = get_feature_target(app_train, app_test)
    X_train, X_val, y_train, _ = get_train_val_sets(X_train, y_train)
    train_data, val_data, test_data = preprocess_data(X_train, X_val, X_test)

    assert train_data.shape[0] == 196806
    assert train_data.shape[1] > 240 and train_data.shape[1] < 250
    assert isinstance(train_data, np.ndarray)
    assert val_data.shape[0] == 49202
    assert val_data.shape[1] > 240 and val_data.shape[1] < 250
    assert isinstance(val_data, np.ndarray)
    assert test_data.shape[0] == 61503
    assert test_data.shape[1] > 240 and test_data.shape[1] < 250
    assert isinstance(test_data, np.ndarray)
