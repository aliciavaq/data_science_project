import json
import os
from pathlib import Path
import pandas as pd
import requests
import time
from datetime import date, timedelta

def get_season_game_ids(year: int) -> list[int]:
    """
    Retrieve all NHL regular-season and playoff game IDs for a season.

    Args:
        year: Starting year of the season (example : 2023 for 2023-24).

    Returns:
        A sorted list of all game IDs for regular-season and playoff games.
    """

    start_date = date(year, 9, 1)        # Start at 01-09 of the given year
    end_date = date(year + 1, 10, 31)    # End at 31-10 of the following year

    game_ids = set()                     # Empty set to store unique game IDs
    current_date = start_date

    # Reuse a session for HTTP requests
    with requests.Session() as session:
        while current_date <= end_date:
            url = (
                "https://api-web.nhle.com/v1/schedule/"
                f"{current_date.isoformat()}"
            )

            response = session.get(url, timeout=30) # Request to API
            response.raise_for_status()      # Raise error if failed
            schedule = response.json()       # Parse JSON response

            for week in schedule.get("gameWeek", []):
                for game in week.get("games", []):
                    game_id = game.get("id")

                    if (
                        game.get("gameType") in (2, 3)   # 2 = reg, 3 = playoffs
                        and game_id is not None
                        and str(game_id).startswith(str(year))  # Only include  
                    ):                                          # games from the 
                        game_ids.add(game_id)                   # season asked

            # Advance by one week to reduce API requests
            current_date += timedelta(days=7)

    return sorted(game_ids)


def get_hockey_data(year: int, path: str = None) -> pd.DataFrame:
    """
    Download or load NHL play-by-play data for a season.

    Retrieves game IDs using get_season_game_ids(), downloads the raw
    JSON data for each game, and saves each response locally. Existing
    files are reused to avoid unnecessary API requests.

    Make sure you set the HOCKEY_DATA_DIR environment variable on your
    computer before using this function. HOCKEY_DATA_DIR is the path used 
    to store all the json files

    Args:
        year: Starting year of the season (example : 2023 for 2023-24).
        path: Directory where raw JSON files will be stored.

    Returns:
        A DataFrame containing the play-by-play events from all games in
        the specified season. Each row represents an event, with its 
        corresponding game_id.
    """

    # Set the data directory
    if path is None:
        path = os.environ.get("HOCKEY_DATA_DIR")

        if not path:
            raise ValueError(
                "Set the HOCKEY_DATA_DIR environment variable or provide a "
                "path argument."
            )

    path = Path(path)

    # Create the directory if it doesn't exist
    season = f"{year}-{str(year + 1)[-2:]}"
    raw_dir = Path(path) / season
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve IDs for regular-season and playoff games
    print(f"Retrieving NHL data for the {season} season.")
    game_ids = get_season_game_ids(year)
    print(f"Number of games found: {len(game_ids)}")

    # If no game IDs were found, raise an error
    if not game_ids:
        raise ValueError(f"No game IDs found for the {season} season.")

    # List to store DataFrames for each game's events
    all_plays = []


    # Reuse a session for HTTP requests
    with requests.Session() as session:

        for i, game_id in enumerate(game_ids, start=1):
            raw_path = raw_dir / f"{game_id}.json"

            # Load cached data if it already exists
            if raw_path.exists():
                with raw_path.open("r", encoding="utf-8") as file:
                    game_data = json.load(file)

            # Download play-by-play data for this game
            else:
                url = (
                    "https://api-web.nhle.com/v1/gamecenter/"
                    f"{game_id}/play-by-play"
                )
            
                response = session.get(url, timeout=30) # Request to API
                response.raise_for_status()             # Raise error if failed
                game_data = response.json()             # Parse JSON response


                # Check we received requested game's event data before saving it
                if game_data.get("id") != game_id:
                    raise ValueError(f"Unexpected game ID returned" 
                                     f"for {game_id}")
                
                if not isinstance(game_data.get("plays"), list):
                    raise ValueError(f"Missing or invalid plays for "
                                      f"game {game_id}")

                # Save the raw JSON response
                with raw_path.open("w", encoding="utf-8") as file:
                    json.dump(
                        game_data,
                        file,
                        ensure_ascii=False,
                        indent=2
                    )

            # Convert the game's events to a DataFrame
            plays_df = pd.json_normalize(game_data["plays"])
            plays_df["game_id"] = game_id
            all_plays.append(plays_df)

            # Display progress
            if i % 100 == 0 or i == len(game_ids):
                print(f"Processed {i}/{len(game_ids)} games")

            time.sleep(1) # Sleep to avoid hitting API requests limit


    return pd.concat(all_plays, ignore_index=True)


# Execute the function for seasons 2016-17 to 2023-24 and print the number of 
# events only if the script is run directly (not imported as a module)
if __name__ == "__main__":
    for year in range(2016, 2024):
        df = get_hockey_data(year)
        print(
            f"Season {year}-{str(year + 1)[-2:]}: "
            f"{df.shape[0]} events"
        )