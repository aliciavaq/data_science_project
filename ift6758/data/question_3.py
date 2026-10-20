import json
from pathlib import Path
import pandas as pd


# This function extracts player names from the game data
def get_player_names(game_data):
    player_names = {}

    # Loop through roster spots to get player IDs and names
    for player in game_data.get("rosterSpots", []):
        player_id = player.get("playerId")

        first_name = player.get("firstName", {})
        last_name = player.get("lastName", {})
       
        # Checks if first_name and last_name are dictionaries and extracts the "default" value if they are. 
        if isinstance(first_name, dict):
            first_name = first_name.get("default")
        if isinstance(last_name, dict):
            last_name = last_name.get("default")

        # Store the full name if the player ID is available
        if player_id is not None:
            player_names[player_id] = " ".join(
                name for name in [first_name, last_name] if name
            )

    return player_names


# This function determines numerical strength from the situationCode 
def get_strength(
    situation_code, event_team_id, home_team_id, away_team_id
):
    if situation_code is None or event_team_id is None:
        return None, None

    # Convert the code to a string and preserve leading zeros (https://docs.python.org/3/builtins/stdtypes.html#str.zfill)
    code = str(situation_code).zfill(4)

    if len(code) != 4 or not code.isdigit():
        return None, None

    # Extract the number of skaters for each team
    away_skaters = int(code[1])
    home_skaters = int(code[2])

    # Identify the event team's number of skaters
    if event_team_id == home_team_id:
        team_skaters = home_skaters
        opponent_skaters = away_skaters
    elif event_team_id == away_team_id:
        team_skaters = away_skaters
        opponent_skaters = home_skaters
    else:
        return None, None

    # Classifying the situation based on the number of skaters
    if team_skaters == opponent_skaters:
        strength_type = "Even strength"
    elif team_skaters > opponent_skaters:
        strength_type = "Power play"
    else:
        strength_type = "Shorthanded"

    # Creating a string showing the number of skaters on each team, from the perspective of the team taking the shot or scoring the goal.
    strength = f"{team_skaters}v{opponent_skaters}"

    return strength_type, strength


# This function creates a DataFrame containing shots on goal and goals for one season
def create_shot_dataframe(year, path):
    # Locate the season directory
    season = f"{year}-{str(year + 1)[-2:]}"
    season_path = Path(path) / season

    if not season_path.exists():
        raise FileNotFoundError(
            f"Season directory not found: {season_path}"
        )

    # Store event data
    events = []

    # Loop through all JSON files in the season directory
    for json_file in sorted(season_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as file:
            game_data = json.load(file)

        game_id = game_data.get("id")

        # Determine the game type from the game ID
        game_type_code = str(game_id)[4:6]

        if game_type_code == "02":
            game_type = "Regular season"
        elif game_type_code == "03":
            game_type = "Playoffs"
        else:
            continue

        player_names = get_player_names(game_data)

        home_team = game_data.get("homeTeam", {})
        away_team = game_data.get("awayTeam", {})

        # Loop through all events in the game
        for play in game_data.get("plays", []):
            event_type = play.get("typeDescKey")

            # Keep shots on goal and goals only
            if event_type not in ["shot-on-goal", "goal"]:
                continue

            details = play.get("details", {})
            period_info = play.get("periodDescriptor", {})

            # Get shooter ID
            shooter_id = details.get("shootingPlayerId")

            if event_type == "goal":
                shooter_id = details.get(
                    "scoringPlayerId", shooter_id
                )

            # Get goalie ID and event team ID
            goalie_id = details.get("goalieInNetId")
            team_id = details.get("eventOwnerTeamId")

            # Match the event team ID to its abbreviation
            team_name = None

            if team_id == home_team.get("id"):
                team_name = home_team.get("abbrev")
            elif team_id == away_team.get("id"):
                team_name = away_team.get("abbrev")

            # Get the situation code and calculate strength
            situation_code = play.get("situationCode")

            strength_type, strength = get_strength(
                situation_code,
                team_id,
                home_team.get("id"),
                away_team.get("id")
            )

            # Store the event information
            events.append({
                "game_id": game_id,
                "event_id": play.get("eventId"),
                "season": season,
                "game_type": game_type,
                "period": period_info.get("number"),
                "time": play.get("timeInPeriod"),
                "team": team_name,
                "event_type": event_type,
                "x": details.get("xCoord"),
                "y": details.get("yCoord"),
                "shooter": player_names.get(shooter_id),
                "goalie": player_names.get(goalie_id),
                "shot_type": details.get("shotType"),
                "is_empty_net": details.get("emptyNet"),
                "situation_code": situation_code,
                "strength_type": strength_type,
                "strength": strength
            })

    # Convert the event list into a DataFrame
    df = pd.DataFrame(events)

    # Check that each event is uniquely identified
    if not df.empty:
        if df.duplicated(["game_id", "event_id"]).any():
            raise ValueError("Duplicate event identifiers found.")

    return df

# Path to directory containing question_3.py
if __name__ == "__main__":
    # Path to directory containing question_3.py
    DATA_DIR = Path(__file__).resolve().parent

    # Creating the DataFrame and show the first 10 rows
    df = create_shot_dataframe(2023, path=DATA_DIR)

    print(df.head(10))

# References : 
# Pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# Python Pathlib: https://docs.python.org/3/library/pathlib.html
# Python JSON: https://docs.python.org/3/library/json.html
# Python (dict.get()) : https://docs.python.org/3/builtins/stdtypes.html#dict.get