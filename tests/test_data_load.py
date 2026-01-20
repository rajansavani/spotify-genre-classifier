import pandas as pd

from src.config import AUDIO_FEATURE_COLS, TARGET_COL
from src.data_load import load_songs

def test_load_songs_minimal_schema(sample_n):
    usecols = ["id", TARGET_COL, "lyrics", *AUDIO_FEATURE_COLS]
    df = load_songs(usecols=usecols, nrows=sample_n)

    assert len(df) > 0
    assert "id" in df.columns
    assert TARGET_COL in df.columns
    assert "lyrics" in df.columns

    # ids should exist and be unique in a small slice
    assert df["id"].isna().sum() == 0
    assert df["id"].duplicated().sum() == 0

def test_audio_cols_are_numeric(sample_n):
    usecols = [TARGET_COL, *AUDIO_FEATURE_COLS]
    df = load_songs(usecols=usecols, nrows=sample_n)

    for c in AUDIO_FEATURE_COLS:
        assert c in df.columns
        assert pd.api.types.is_numeric_dtype(df[c])