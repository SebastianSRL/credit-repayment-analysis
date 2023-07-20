import pandas as pd

from src.data_utils import get_datasets, get_feature_target, get_train_val_sets


def test_get_datasets():
    """Tests for get_datasets() function."""
    _app_train, _app_test, _columns_description = get_datasets()

    assert _app_train.shape == (246008, 122)
    assert isinstance(_app_train, pd.DataFrame)
    assert _app_test.shape == (61503, 122)
    assert isinstance(_app_test, pd.DataFrame)
    assert len(_columns_description) == 122


def test_get_feature_target(app_train: pd.DataFrame, app_test: pd.DataFrame):
    """Tests for get_feature_target() function."""
    X_train, y_train, X_test, y_test = get_feature_target(app_train, app_test)

    assert X_train.shape == (246008, 121)
    assert isinstance(X_train, pd.DataFrame)
    assert y_train.shape == (246008,) or y_train.shape == (246008, 1)
    assert X_test.shape == (61503, 121)
    assert isinstance(X_test, pd.DataFrame)
    assert y_test.shape == (61503,) or y_test.shape == (61503, 1)


def test_get_train_val_sets(app_train: pd.DataFrame, app_test: pd.DataFrame):
    """Tests for get_train_val_sets() function."""
    X_train, y_train, _, _ = get_feature_target(app_train, app_test)
    X_train, X_val, y_train, y_val = get_train_val_sets(X_train, y_train)

    assert X_train.shape == (196806, 121)
    assert isinstance(X_train, pd.DataFrame)
    assert y_train.shape == (196806,) or y_train.shape == (196806, 1)
    assert X_val.shape == (49202, 121)
    assert isinstance(X_val, pd.DataFrame)
    assert y_val.shape == (49202,) or y_val.shape == (49202, 1)
