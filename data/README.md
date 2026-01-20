# Data

This project uses the kaggle dataset below (songs + artists):

- https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres/data?select=songs.csv

## Expected folder layout

    data/
      raw/
        songs.csv
        artists.csv
      processed/
        (generated files go here)

## Raw data (not tracked in git)

Place the downloaded csv files here:

- `data/raw/songs.csv`
- `data/raw/artists.csv`

These files are ignored by git to keep the repo lightweight.

## Processed data (generated)

`data/processed/` is where scripts can write cleaned datasets, cached splits, or feature files.
Right now it may be empty, depending on what you run.
