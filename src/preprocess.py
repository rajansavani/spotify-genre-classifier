from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_audio_pipeline() -> Pipeline:
    """
    Preprocessing for spotify-style numeric audio features.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), # robust to outliers
            ("scaler", StandardScaler()), # helps linear models converge + behave well
        ]
    )


def make_audio_preprocessor(audio_cols: list[str]) -> ColumnTransformer:
    """
    Wraps the audio pipeline in a column transformer.
    Useful for audio-only models.
    """
    audio_pipe = make_audio_pipeline()

    return ColumnTransformer(
        transformers=[
            ("audio", audio_pipe, audio_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )