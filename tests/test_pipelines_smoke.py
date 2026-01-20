from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import AUDIO_FEATURE_COLS, TARGET_COL
from src.data_load import load_songs
from src.preprocess import make_audio_preprocessor
from sklearn.compose import ColumnTransformer


def test_lyrics_pipeline_fit_predict(sample_n):
    df = load_songs(usecols=[TARGET_COL, "lyrics"], nrows=sample_n)
    df["lyrics"] = df["lyrics"].fillna("").astype(str)

    X = df["lyrics"]
    y = df[TARGET_COL]

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X.iloc[:50])

    assert len(preds) == 50


def test_fusion_pipeline_fit_predict(sample_n):
    df = load_songs(usecols=[TARGET_COL, "lyrics", *AUDIO_FEATURE_COLS], nrows=sample_n)
    df["lyrics"] = df["lyrics"].fillna("").astype(str)

    X = df[["lyrics", *AUDIO_FEATURE_COLS]]
    y = df[TARGET_COL]

    pre = ColumnTransformer(
        transformers=[
            ("lyrics", TfidfVectorizer(max_features=5000, stop_words="english"), "lyrics"),
            ("audio", make_audio_preprocessor(AUDIO_FEATURE_COLS), AUDIO_FEATURE_COLS),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("features", pre),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X.iloc[:50])

    assert len(preds) == 50