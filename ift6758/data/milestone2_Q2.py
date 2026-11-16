"""
Reusable dataset helpers for milestone 2
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ift6758.data import create_shot_dataframe


all_years = range(2016, 2023)   # 2016-17 to 2022-23: training and validation
test_year = 2023                # 2023-24: final test set, used only in part 7
split_seed = 42
val_fraction = 0.2


def load_seasons(years, data_dir, cache_path=None):
    """
    Build the shot table for several seasons with create_shot_dataframe

    If cache_path is given, the table is saved there as a CSV the first time
    and read back afterwards (delete the CSV if create_shot_dataframe changes)
    """
    if cache_path is not None and Path(cache_path).exists():
        df = pd.read_csv(cache_path)
    else:
        frames = [create_shot_dataframe(year, data_dir) for year in years]
        df = pd.concat(frames, ignore_index=True)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)

    # CSV does not keep pandas nullable types : restore them
    df["is_empty_net"] = df["is_empty_net"].astype("Int64")
    df["def_zone"] = df["def_zone"].astype("boolean")
    return df


def filter_shots(df, game_type="Regular season", verbose=True):
    """
    Keep one game type and remove shootout attempts
    Prints how many events are excluded so they can be reported
    """
    selected = df[df["game_type"] == game_type].copy()
    is_shootout = selected["period_type"].eq("SO")
    shots = selected[~is_shootout].copy()

    if verbose:
        print(f"{game_type}: {len(df) - len(selected)} events of other game types left out")
        print(f"{game_type}: {int(is_shootout.sum())} shootout attempts excluded")
        print(f"{game_type}: {len(shots)} events kept")
    return shots


def split_train_val(df, val_fraction=val_fraction, seed=split_seed):
    """
    Add a "split" column ("train" or "val"), drawing whole games at random

    All shots of a game go to the same set, because shots from the same game
    are related (same teams, same goalies, rebounds). Game IDs are sorted
    before the draw, so the same data and seed always give the same split
    """
    df = df.copy()
    game_ids = np.sort(df["game_id"].unique())
    rng = np.random.default_rng(seed)
    val_games = rng.choice(game_ids, size=int(val_fraction * len(game_ids)), replace=False)
    df["split"] = np.where(df["game_id"].isin(val_games), "val", "train")
    return df


def load_dev_set(data_dir, cache_dir=None, verbose=True):
    """
    Regular-season shots from 2016-17 to 2022-23 (shootouts excluded),
    with a reproducible train / validation split by game
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "processed"
    all_shots = load_seasons(all_years, data_dir, cache_dir / "shots_2016_2022_all.csv")
    assert test_year not in set(all_shots["season"].str[:4].astype(int)), "Test season loaded by mistake."
    shots = filter_shots(all_shots, "Regular season", verbose=verbose)
    return split_train_val(shots)


def load_test_set(data_dir, game_type="Regular season", cache_dir=None, verbose=True):
    """
    2023-24 shots, processed exactly like the development data
    Use only in part 7
    game_type: "Regular season" or "Playoffs"
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "processed"
    all_shots = load_seasons([test_year], data_dir, cache_dir / "shots_2023_all.csv")
    return filter_shots(all_shots, game_type, verbose=verbose)