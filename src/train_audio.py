from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .config import AUDIO_FEATURE_COLS, PATHS, RANDOM_SEED, TARGET_COL
from .data_load import load_songs
from .preprocess import make_audio_preprocessor


def _ensure_dirs() -> None:
    # create output dirs if they don't exist
    PATHS.models.mkdir(parents=True, exist_ok=True)
    (PATHS.reports / "figures").mkdir(parents=True, exist_ok=True)


def train_audio_baselines(sample_n: int | None = None) -> None:
    """
    Trains two audio-only baselines (predicting genre from audio features):
        1) logistic regression
        2) xgboost

    saves:
        - trained models in models/
        - label encoder in models/
        - confusion matrix plots in reports/figures/
    """
    _ensure_dirs()

    # load only what we need for audio baseline training
    usecols = [TARGET_COL, *AUDIO_FEATURE_COLS]
    df = load_songs(usecols=usecols)

    # optional downsample for faster iteration
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=RANDOM_SEED)

    # features and target
    X = df[AUDIO_FEATURE_COLS]
    y = df[TARGET_COL]

    # encode labels for models that require integer class ids (xgboost)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    # split with stratification due to class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )

    preprocessor = make_audio_preprocessor(AUDIO_FEATURE_COLS)

    # model 1: logistic regression
    lr = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
    )
    lr_pipe = Pipeline(steps=[("prep", preprocessor), ("model", lr)])
    lr_pipe.fit(X_train, y_train)

    lr_pred = lr_pipe.predict(X_test)
    lr_macro_f1 = f1_score(y_test, lr_pred, average="macro")

    print("\nlogistic regression report")
    print(f"macro f1: {lr_macro_f1:.4f}")
    print(classification_report(y_test, lr_pred, target_names=class_names))

    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        lr_pred,
        display_labels=class_names,
        xticks_rotation=45,
        normalize=None,
    )
    plt.title(f"confusion matrix (logreg) | macro f1={lr_macro_f1:.3f}")
    out_path = PATHS.reports / "figures" / "cm_logreg.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    joblib.dump(lr_pipe, PATHS.models / "audio_logreg.joblib")

    # model 2: xgboost
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(class_names),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_pipe = Pipeline(steps=[("prep", preprocessor), ("model", xgb)])
    xgb_pipe.fit(X_train, y_train)

    xgb_pred = xgb_pipe.predict(X_test)
    xgb_macro_f1 = f1_score(y_test, xgb_pred, average="macro")

    print("\nxgboost report")
    print(f"macro f1: {xgb_macro_f1:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=class_names))

    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        xgb_pred,
        display_labels=class_names,
        xticks_rotation=45,
        normalize=None,
    )
    plt.title(f"confusion matrix (xgboost) | macro f1={xgb_macro_f1:.3f}")
    out_path = PATHS.reports / "figures" / "cm_xgboost.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    joblib.dump(xgb_pipe, PATHS.models / "audio_xgboost.joblib")

    # save the label encoder so inference can map ints -> genre strings
    joblib.dump(le, PATHS.models / "label_encoder.joblib")

    print("\nsaved models to:", PATHS.models)
    print("saved figures to:", PATHS.reports / "figures")


if __name__ == "__main__":
    train_audio_baselines(sample_n=None)
