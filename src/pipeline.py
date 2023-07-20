# Purpose: Pipeline for preprocessing data
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.transformers import OutliersHandler


def get_pipeline(data: pd.DataFrame, threshold: int = 365243):
    obj = dict(one_hot=list(), binary=list())
    for col in data.select_dtypes(include="object").columns:
        if data[col].nunique() == 2:
            obj["binary"].append(col)
        else:
            obj["one_hot"].append(col)

    transforms = ColumnTransformer(
        [
            ("outliers", OutliersHandler(threshold), ["DAYS_EMPLOYED"]),
            ("ordinal", OrdinalEncoder(), obj["binary"]),
            ("one_hot", OneHotEncoder(sparse_output=False), obj["one_hot"]),
        ],
        remainder="passthrough",
    )

    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

    model = LGBMClassifier()

    pipeline = Pipeline(
        [
            ("transforms", transforms),
            ("imputer", imputer),
            ("scaler", scaler),
            (type(model).__name__, model),
        ]
    )

    return pipeline
