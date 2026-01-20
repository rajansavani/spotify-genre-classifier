import numpy as np

from src.config import AUDIO_FEATURE_COLS, TARGET_COL
from src.data_load import load_songs
from src.preprocess import make_audio_preprocessor


def test_audio_preprocessor_output_shape(sample_n):
    usecols = [TARGET_COL, *AUDIO_FEATURE_COLS]
    df = load_songs(usecols=usecols, nrows=sample_n)

    X = df[AUDIO_FEATURE_COLS]
    pre = make_audio_preprocessor(AUDIO_FEATURE_COLS)

    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] == len(AUDIO_FEATURE_COLS)
    assert np.isfinite(Xt).all()
