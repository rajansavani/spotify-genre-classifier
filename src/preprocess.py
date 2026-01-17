from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_audio_preprocessor(audio_cols: list[str]) -> ColumnTransformer:
    # numeric pipeline for spotify-style audio features
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("audio", numeric_pipe, audio_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
