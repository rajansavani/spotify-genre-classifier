from __future__ import annotations

import argparse
from typing import Optional

import joblib
import pandas as pd

from .config import AUDIO_FEATURE_COLS, PATHS


MODEL_FILES = {
    "audio_logreg": "audio_logreg.joblib",
    "audio_xgboost": "audio_xgboost.joblib",
    "lyrics_logreg": "lyrics_logreg.joblib",
    "fusion_logreg": "fusion_logreg.joblib",
}

def _load_model(model_name: str):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILES)}")

    model_path = PATHS.models / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at: {model_path}")
    
    return joblib.load(model_path)

def _load_label_encoder():
    # we save one encoder file
    # it should be consistent across runs since genre set is fixed
    enc_path = PATHS.models / "label_encoder.joblib"
    if not enc_path.exists():
        raise FileNotFoundError(f"Label encoder not found at: {enc_path}. Run a train script that saves it.")
    return joblib.load(enc_path)

def _find_song_by_id(song_id: str, usecols: list[str], chunksize: int = 200_000) -> pd.DataFrame:
    """
    Chunk-scan songs.csv for a single spotify id.
    Returns a single-row dataframe if found, else raises.
    """
    csv_path = PATHS.songs_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"songs.csv not found at: {csv_path}")
    
    # always include id for filtering
    cols = ["id", *[c for c in usecols if c != "id"]]

    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize):
        hit = chunk[chunk["id"] == song_id]
        if not hit.empty:
            return hit.iloc[[0]].copy()
    
    raise ValueError(f"Song id '{song_id}' not found in songs.csv")

def _topk_from_proba(proba, class_names: list[str], k: int = 5):
    proba = proba.ravel()
    k = max(1, min(k, len(proba)))
    idx = proba.argsort()[::-1][:k]
    return [(class_names[i], float(proba[i])) for i in idx]

def _predict_audio(model, le, song_row: pd.DataFrame, top_k: int) -> None:
    # model expects a dataframe with audio feature columns
    X = song_row[AUDIO_FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    top = _topk_from_proba(proba, list(le.classes_), k=top_k)

    print("\nInput: audio features (from songs.csv)")
    print("Top predictions:")
    for label, p in top:
        print(f"  {label:<10}  {p:.4f}")

def _predict_lyrics(model, le, lyrics_text: str, top_k: int) -> None:
    # lyrics pipeline was trained on raw text sequences (not a dataframe)
    proba = model.predict_proba([lyrics_text])[0]
    top = _topk_from_proba(proba, list(le.classes_), k=top_k)

    print("\nInput: lyrics text")
    print("Top predictions:")
    for label, p in top:
        print(f"  {label:<10}  {p:.4f}")

def _predict_fusion(model, le, lyrics_text: str, song_row: Optional[pd.DataFrame], top_k: int) -> None:
    # fusion pipeline expects a dataframe with lyrics + audio columns
    if song_row is None:
        # if user only provides text, we fill audio with nan
        # imputer will use medians learned in training
        row = {c: [float("nan")] for c in AUDIO_FEATURE_COLS}
        row["lyrics"] = [lyrics_text]
        X = pd.DataFrame(row)
        source = "lyrics text + default audio (median-imputed)"
    else:
        song_row = song_row.copy()
        song_row["lyrics"] = song_row["lyrics"].fillna("").astype(str)
        X = song_row[["lyrics", *AUDIO_FEATURE_COLS]]
        source = "lyrics + audio (from songs.csv)"

    proba = model.predict_proba(X)[0]
    top = _topk_from_proba(proba, list(le.classes_), k=top_k)

    print(f"\nInput: fusion ({source})")
    print("Top predictions:")
    for label, p in top:
        print(f"  {label:<10}  {p:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="predict spotify genre using trained models")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_FILES.keys()),
        help="which trained model to use",
    )
    parser.add_argument("--song-id", type=str, default=None, help="spotify track id from songs.csv")
    parser.add_argument("--text", type=str, default=None, help="raw lyrics text")
    parser.add_argument("--top-k", type=int, default=5, help="number of top predictions to print")

    args = parser.parse_args()

    if args.song_id is None and args.text is None:
        raise ValueError("Must provide at least one of --song-id or --text")
    
    model = _load_model(args.model)
    le = _load_label_encoder()

    if args.model.startswith("audio_"):
        if args.song_id is None:
            raise ValueError("Audio model requires --song-id to load audio features from songs.csv")
        song_row = _find_song_by_id(args.song_id, usecols=["id", *AUDIO_FEATURE_COLS])
        _predict_audio(model, le, song_row, top_k=args.top_k)
        return
    
    if args.model == "lyrics_logreg":
        if args.text is None and args.song_id is not None:
            song_row = _find_song_by_id(args.song_id, usecols=["id", "lyrics"])
            lyrics_text = str(song_row["lyrics"].iloc[0]) if pd.notna(song_row["lyrics"].iloc[0]) else ""
        else:
            lyrics_text = args.text or ""
        _predict_lyrics(model, le, lyrics_text, top_k=args.top_k)
        return

    if args.model == "fusion_logreg":
        song_row = None
        lyrics_text = args.text or ""

        if args.song_id is not None:
            song_row = _find_song_by_id(args.song_id, usecols=["id", "lyrics", *AUDIO_FEATURE_COLS])
            # if user didn't provide --text, take lyrics from the dataset row
            if args.text is None:
                lyrics_text = str(song_row["lyrics"].iloc[0]) if pd.notna(song_row["lyrics"].iloc[0]) else ""

        _predict_fusion(model, le, lyrics_text, song_row, top_k=args.top_k)
        return

    raise ValueError(f"Unhandled model type: {args.model}")

if __name__ == "__main__":
    main()