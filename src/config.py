from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    # project root = repo root (src/ one level below)
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    models: Path = root / "models"
    reports: Path = root / "reports"

    songs_csv: Path = data_raw / "songs.csv"
    artists_csv: Path = data_raw / "artists.csv"

PATHS = Paths()

# random seed used for splits + reproducibility
RANDOM_SEED = 1738

# audio feature columns (as per Spotify API)
AUDIO_FEATURE_COLS = [
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
]

# target column for the first pass of this project
TARGET_COL = "genre"

# columns that are "list-like" (but stored as strings in the csvs)
SONGS_LIST_COLS = ["artists", "artist_ids", "niche_genres"]
ARTISTS_LIST_COLS = ["genres"]

# a minimal set of columns for EDA
SONGS_EDA_COLS = [
    "id",
    "name",
    "album_name",
    "artists",
    "year",
    "genre",
    "popularity",
    "total_artist_followers",
    "avg_artist_popularity",
    "artist_ids",
    "niche_genres",
    "lyrics",
    *AUDIO_FEATURE_COLS,
]

ARTISTS_EDA_COLS = [
    "id",
    "name",
    "followers",
    "popularity",
    "genres",
    "main_genre",
]