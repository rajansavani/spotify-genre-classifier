from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import PATHS, RANDOM_SEED, TARGET_COL
from .data_load import load_songs

def _ensure_dirs() -> None:
    # create output dirs if they don't exist
    PATHS.models.mkdir(parents=True, exist_ok=True)
    (PATHS.reports / "figures").mkdir(parents=True, exist_ok=True)

def train_lyrics_baseline(sample_n: int | None = 200_000) -> None:
    """
    Trains a lyrics-only genre classifier using tf-idf + logistic regression.

    saves:
        - trained model in models/
        - label encoder in models/
        - confusion matrix plot in reports/figures/
    """
    _ensure_dirs()

    # load only what we need
    usecols = [TARGET_COL, "lyrics"]
    df = load_songs(usecols=usecols)

    # optional downsample for faster iteration
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=RANDOM_SEED)

    # basic text cleanup: handle missing / non-string
    df["lyrics"] = df["lyrics"].fillna("").astype(str)

    X_text = df["lyrics"]
    y = df[TARGET_COL]

    # encode labels for reporting + saving
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )

    # tf-idf vectorizer config:
    # - unigrams + bigrams
    # - max_features caps vocab size to control memory
    # - min_df drops very rare tokens (noise)
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=100_000,
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("tfidf", tfidf), ("model", clf)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nlyrics (tf-idf + logreg) report")
    print(f"macro f1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    
    # confusion matrix plot
    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        normalize=None,
    )
    plt.title(f"confusion matrix (lyrics logreg) | macro f1={macro_f1:.3f}")
    out_path = PATHS.reports / "figures" / "cm_lyrics_logreg.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    # save model + encoder
    joblib.dump(pipe, PATHS.models / "lyrics_logreg.joblib")
    joblib.dump(le, PATHS.models / "label_encoder.joblib")

    print("\nsaved model to:", PATHS.models / "lyrics_logreg.joblib")
    print("saved figure to:", out_path)


if __name__ == "__main__":
    train_lyrics_baseline(sample_n=None)