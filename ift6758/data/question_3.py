import json
from pathlib import Path
import pandas as pd

# This function is used to extract player names from the game data
def get_player_names(game_data):
    player_names = {}
    # Loop through the roster spots to get player IDs and names
    for player in game_data.get("rosterSpots", []):
        player_id = player.get("playerId")

        first_name = player.get("firstName", {})
        last_name = player.get("lastName", {})

        if isinstance(first_name, dict):
            first_name = first_name.get("default")

        if isinstance(last_name, dict):
            last_name = last_name.get("default")
        
        # Stores the full name in the dictionary if player_id is not None
        if player_id is not None:
            player_names[player_id] = " ".join(
                name for name in [first_name, last_name] if name
            )
    return player_names

# This function is used to create a DataFrame containing shots on goal and goals for one NHL season. 
# Each row represents one event. 
def create_shot_dataframe(year, path):
    # Locate the season directory
    season = f"{year}-{str(year + 1)[-2:]}"
    season_path = Path(path) / season

    if not season_path.exists():
        raise FileNotFoundError(
            f"Season directory not found: {season_path}"
        )
    
    # Creates a list to store event data
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
            game_type = None

        player_names = get_player_names(game_data)

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

            # Get goalie ID
            goalie_id = details.get("goalieInNetId")

            # Get the ID of the team responsible for the event
            team_id = details.get("eventOwnerTeamId")

            home_team = game_data.get("homeTeam", {})
            away_team = game_data.get("awayTeam", {})

            team_name = None

            # Match the event's team ID to the correct team
            if team_id is not None:
                if team_id == home_team.get("id"):
                    team_name = home_team.get("abbrev")

                elif team_id == away_team.get("id"):
                    team_name = away_team.get("abbrev")

            # Adding the event data to the list of events. 
            # Missing values remain missing instead of being replaced.
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
                "strength": details.get("strength"),
            })
    
    # Convert the list into a Pandas DataFrame.   
    df = pd.DataFrame(events)
    # Check for duplicate event identifiers in the DataFrame. 
    # If duplicates are found, raise a ValueError.
    if not df.empty:
        if df.duplicated(["game_id", "event_id"]).any():
            raise ValueError("Duplicate event identifiers found.")

    return df