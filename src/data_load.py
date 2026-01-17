from __future__ import annotations

import ast
import json
from typing import Any, Iterable, Optional

import pandas as pd

from .config import (
    PATHS,
    ARTISTS_EDA_COLS,
    ARTISTS_LIST_COLS,
    SONGS_EDA_COLS,
    SONGS_LIST_COLS,
)

def _safe_parse_list(value: Any) -> list:
    """
    Parse a cell that should represent a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() == "nan":
            return []
        # if it doesn't look like a list, treat as empty
        if not (s.startswith("[") and s.endswith("]")):
            return []

        # try json first (["a", "b", "c"])
        try:
            return json.loads(s)
        except Exception:
            pass

        # some rows look like python literals (['a', 'b', 'c'])
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
        
    return []

def _coerce_list_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_safe_parse_list)
    return df

def load_artists(
    path: Optional[str] = None,
    usecols: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load artists.csv with consistent types.
    """
    csv_path = PATHS.artists_csv if path is None else path
    if usecols is None:
        usecols = ARTISTS_EDA_COLS
    
    df = pd.read_csv(csv_path, usecols=usecols, nrows=nrows)

    # basic type cleanup
    if "followers" in df.columns:
        df["followers"] = pd.to_numeric(df["followers"], errors="coerce")
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    df = _coerce_list_cols(df, ARTISTS_LIST_COLS)
    return df

def load_songs(
    path: Optional[str] = None,
    usecols: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load songs.csv with consistent types. Lyrics can be large, so usecols lets you trim memory during EDA.
    """
    csv_path = PATHS.songs_csv if path is None else path
    if usecols is None:
        usecols = SONGS_EDA_COLS

    df = pd.read_csv(csv_path, usecols=usecols, nrows=nrows)

    # basic type cleanup
    numeric_cols = [
        c
        for c in df.columns
        if c
        in {
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "year",
            "popularity",
            "total_artist_followers",
            "avg_artist_popularity",
        }
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _coerce_list_cols(df, SONGS_LIST_COLS)
    return df

def basic_sanity_report(songs: pd.DataFrame, artists: Optional[pd.DataFrame] = None) -> dict:
    report: dict[str, Any] = {}

    report["songs_rows"] = int(len(songs))
    report["songs_cols"] = list(songs.columns)
    if "id" in songs.columns:
        report["songs_id_nulls"] = int(songs["id"].isna().sum())
        report["songs_id_duplicates"] = int(songs["id"].duplicated().sum())

    if "genre" in songs.columns:
        report["genre_value_counts"] = songs["genre"].value_counts(dropna=False).head(15).to_dict()

    # range checks for spotify-style features
    bounded_01 = [c for c in ["danceability", "energy", "speechiness", "acousticness",
                             "instrumentalness", "liveness", "valence"] if c in songs.columns]
    for c in bounded_01:
        col = songs[c].dropna()
        report[f"{c}_out_of_range"] = int(((col < 0.0) | (col > 1.0)).sum())

    if "tempo" in songs.columns:
        col = songs["tempo"].dropna()
        report["tempo_nonpositive"] = int((col <= 0).sum())

    if "duration_ms" in songs.columns:
        col = songs["duration_ms"].dropna()
        report["duration_too_short_<10s"] = int((col < 10_000).sum())
        report["duration_too_long_>30min"] = int((col > 1_800_000).sum())

    if artists is not None:
        report["artists_rows"] = int(len(artists))
        if "id" in artists.columns:
            report["artists_id_nulls"] = int(artists["id"].isna().sum())
            report["artists_id_duplicates"] = int(artists["id"].duplicated().sum())

    return report