from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import AUDIO_FEATURE_COLS, PATHS, RANDOM_SEED, TARGET_COL
from .data_load import load_songs
from .preprocess import make_audio_pipeline

def _ensure_dirs() -> None:
    # create output dirs if they don't exist
    PATHS.models.mkdir(parents=True, exist_ok=True)
    (PATHS.reports / "figures").mkdir(parents=True, exist_ok=True)

def train_fusion_logreg(sample_n: int | None = None) -> None:
    """
    Trains a fusion model using:
        - numeric spotify audio features
        - tf-idf features from lyrics

    Model: logistic regression
    """
    _ensure_dirs()

    usecols = [TARGET_COL, "lyrics", *AUDIO_FEATURE_COLS]
    df = load_songs(usecols=usecols)

    # optional downsample for faster iteration
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=RANDOM_SEED)
    
    # tf-idf expects strings
    df["lyrics"] = df["lyrics"].fillna("").astype(str)

    X = df[["lyrics", *AUDIO_FEATURE_COLS]]
    y = df[TARGET_COL]

    # encode labels for reporting + saving
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )

    # audio preprocessor from preprocess.py
    audio_pipe = make_audio_pipeline()

    # text featurizer
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=100_000,
    )

    # combine dense audio features + sparse text features
    preprocessor = ColumnTransformer(
        transformers=[
            ("audio", audio_pipe, AUDIO_FEATURE_COLS),
            ("lyrics", tfidf, "lyrics"),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nfusion (audio + lyrics) report")
    print(f"macro f1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
    )
    plt.title(f"confusion matrix (fusion logreg) | macro f1={macro_f1:.3f}")
    out_path = PATHS.reports / "figures" / "cm_fusion_logreg.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    joblib.dump(pipe, PATHS.models / "fusion_logreg.joblib")
    joblib.dump(le, PATHS.models / "label_encoder.joblib")

    print("\nsaved model to:", PATHS.models / "fusion_logreg.joblib")
    print("saved figure to:", out_path)


if __name__ == "__main__":
    train_fusion_logreg(sample_n=None)