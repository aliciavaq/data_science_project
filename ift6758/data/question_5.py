import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

from ift6758.data import *


# Get the hockey data directory from Question 1
hockey_data_dir = os.environ.get("HOCKEY_DATA_DIR")

# Use the specified directory (environment variable)
DATA_DIR = Path(hockey_data_dir)
if not DATA_DIR.exists():
    raise FileNotFoundError("Hockey data directory not found.")

# Define the path to the rink image
rink_path =  "figures/nhl_rink.png"

# Define the bin edges for the x and y coordinates of the rink
x_edges = np.arange(25, 101, 5)
y_edges = np.arange(-42.5, 43, 5)

# Output directory for the interactive HTML files
output_dir = Path("figures/visualizations")


def normalize_shot_coordinates(shots):
    """
    Orient shots so x is positive and flip y when x was negative.

    Returns a new DataFrame with the normalized coordinates added as columns.

    Args:
        shots: DataFrame containing shot events with their x and y coordinates.
    """
    # Create a copy of the shots DataFrame to avoid modifying the original
    normalized = shots.copy()

    # Normalize x and y coordinates
    normalized["x_normalized"] = normalized["x"].abs()

    # Flip y-coordinate for shots taken from the left side of the rink
    normalized["y_normalized"] = np.where(
        normalized["x"] < 0,
        -normalized["y"],
        normalized["y"]
    )

    return normalized



def count_shots_per_bin(shots, x_edges, y_edges):
    """
    Count shots in each spatial bin of the rink. 

    Returns an array with the count of shots in each bin.

    Args:
        shots: DataFrame containing shot events with normalized x and y coords.
        x_edges: Array of bin edges for the x-axis (length = # of bins + 1).
        y_edges: Array of bin edges for the y-axis (length = # of bins + 1).
    """
    # Count shots in each bin 
    counts, _, _ = np.histogram2d(
        shots["x_normalized"],
        shots["y_normalized"],
        bins=[x_edges, y_edges]
    )

    return counts



def calculate_team_bin_counts(shots, x_edges, y_edges):
    """
    Count shots in every spatial bin for every team.

    Returns a DataFrame with one row per team and bin.

    Args:
        shots: DataFrame containing shot events with normalized x and y coords.
        x_edges: Array of bin edges for the x-axis (length = # of bins + 1).
        y_edges: Array of bin edges for the y-axis (length = # of bins + 1).
    """
    # Initialize a list to store the results
    results = []

    # Group shots by team and count shots in each bin
    for team, team_shots in shots.groupby("team"):
        counts = count_shots_per_bin(team_shots, x_edges, y_edges)

        # Go through each bin and store the counts along with the team name
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                # Store the team name, bin indices, bin centers, and shot counts
                results.append({
                    "team": team,
                    "x_bin": i,
                    "y_bin": j,
                    "x_center": (x_edges[i] + x_edges[i + 1]) / 2,
                    "y_center": (y_edges[j] + y_edges[j + 1]) / 2,
                    "shots": counts[i, j]
                })

    return pd.DataFrame(results)



def count_games_per_team(shots, data_dir):
    """
    Count how many games each team played, using the JSONs.

    Returns a DataFrame with one row per team and the number of games played.

    Args:
        shots: DataFrame containing shot events with game IDs and team names.
        data_dir: Directory where the raw JSON files are stored.
    """
    # Get the season from the first shot's game ID
    season_dir = Path(data_dir) / shots["season"].iloc[0]

    # Initialize a list to store the game counts for each team
    games = []

    # Get unique game IDs from the shots DataFrame
    game_ids = shots["game_id"].drop_duplicates().tolist()

    # Loop through each game ID and check if the JSON file exists
    for game_id in game_ids:
        json_path = season_dir / f"{game_id}.json"

        # If the JSON file doesn't exist, skip this game and print a warning
        if not json_path.exists():
            print(f"Missing game file: {json_path}")
            continue

        # Load the JSON data for the game to get the home and away teams
        with json_path.open("r", encoding="utf-8") as file:
            game_data = json.load(file)

        # Get the home and away team abbreviations from the JSON data
        home_team = game_data.get("homeTeam", {}).get("abbrev")
        away_team = game_data.get("awayTeam", {}).get("abbrev")

        # If both teams are found, add them to the list of games
        if home_team and away_team:
            games.append({"game_id": game_id, "team": home_team})
            games.append({"game_id": game_id, "team": away_team})

    games_df = pd.DataFrame(games)

    # Group by team and count unique game IDs to get the number of games played.
    # Add the "games_played" column to the DataFrame and return it.
    return (
        games_df.groupby("team")["game_id"]
        .nunique()
        .reset_index(name="games_played")
    )


def calculate_league_shot_rates(shots, games_per_team, x_edges, y_edges):
    """
    Calculate league-average shots per team-game for each spatial bin.

    Returns a DataFrame with one row per bin, containing the league-average
    shot rate per team-game and the bin's center coordinates.

    Args:
        shots: DataFrame containing shot events with normalized x and y coords.
        games_per_team: DataFrame containing the # of games played by each team.
        x_edges: Array of bin edges for the x-axis (length = # of bins + 1).
        y_edges: Array of bin edges for the y-axis (length = # of bins + 1).
    """
    # Count shots in each spatial bin across the entire league
    league_counts = count_shots_per_bin(shots, x_edges, y_edges)

    # Calculate the total number of team-games in the league
    total_team_games = games_per_team["games_played"].sum()

    # Create a list to store the results for each bin
    results = []

    # Iterate through each bin
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):

            # Store the league shot rate per team-game and the bin's center 
            # coordinates
            results.append({
                "x_bin": i,
                "y_bin": j,
                "league_shots": league_counts[i, j],
                "league_shots_per_team_game": (
                    league_counts[i, j] / total_team_games
                ),
                "x_center": (x_edges[i] + x_edges[i + 1]) / 2,
                "y_center": (y_edges[j] + y_edges[j + 1]) / 2
            })

    return pd.DataFrame(results)


def calculate_team_shot_rates(team_bin_counts, games_per_team):
    """
    Calculate shots per team-hour for each team and spatial bin.

    Returns a DataFrame with one row per team and bin, including the shot rate 
    per team-hour.

    Args:
        team_bin_counts: DataFrame containing shot counts for each team and bin.
        games_per_team: DataFrame containing the # of games played by each team.
    """

    # Add the number of games played by each team to its bin counts
    rates = team_bin_counts.merge(
        games_per_team,
        on="team",
        how="left"
    )

    # Calculate the shot rate per team-hour
    rates["shots_per_team_game"] = (
        rates["shots"] / rates["games_played"]
    )

    return rates



def calculate_team_league_differences(team_shot_rates, league_shot_rates):
    """
    Calculate each team's shot rate minus the league rate in each bin.

    Returns a DataFrame with one row per team and bin, containing the difference
    between the team's shot rate and the league-average shot rate, along with 
    the bin's center coordinates.

    Args:
        team_shot_rates: DataFrame containing each team's shot rate per bin.
        league_shot_rates: DataFrame containing the league-average shot rate
        per bin.
    """
    # Merge the team shot rates with the league shot rates for each bin
    differences = team_shot_rates.merge(
        league_shot_rates[
            ["x_bin", "y_bin", "league_shots_per_team_game"]
        ],
        on=["x_bin", "y_bin"],
        how="left"
    )

    # Calculate the difference between each team's shot rate and the league-
    # average shot rate for each bin
    differences["shot_rate_difference"] = (
        differences["shots_per_team_game"]
        - differences["league_shots_per_team_game"]
    )

    return differences



def ensure_season_game_files(year, data_dir):
    """
    Check whether all expected NHL game JSON files exist for a season.
    If files are missing, use get_hockey_data() function to download them.

    Returns nothing.

    Args:
        year: Starting year of the season.
        data_dir: Root directory containing the season folders.
    """

    # Locate the season directory
    season = f"{year}-{str(year + 1)[-2:]}"
    season_dir = Path(data_dir) / season
    season_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve all expected game IDs for the season
    expected_game_ids = get_season_game_ids(year)

    if not expected_game_ids:
        raise ValueError(
            f"No game IDs found for season {season}."
        )

    # Find game IDs whose JSON files are missing
    missing_game_ids = [
        game_id
        for game_id in expected_game_ids
        if not (season_dir / f"{game_id}.json").is_file()
    ]

    print(
        f"Season {season}: {len(expected_game_ids)} games expected, "
        f"{len(missing_game_ids)} files missing."
    )

    # Download missing files if necessary
    if missing_game_ids:
        print(f"Retrieving missing game data for season {season}...")

        # get_hockey_data reuses existing JSON files and downloads missing ones
        get_hockey_data(
            year=year,
            path=str(data_dir),
            game_ids=missing_game_ids
        )

    else:
        print(f"All game files are already present for season {season}.")


    print(f"All expected game files are available for season {season}.")



def create_shot_df(year, data_dir):
    """
    Ensure all expected game JSON files exist, then create the shot DataFrame.

    Returns a DataFrame containing all shot events for the season, with 
    normalized coordinates and the corresponding game_id.

    Args:
        year: Starting year of the season.
        data_dir: Directory where the raw JSON files are stored.
    """

    ensure_season_game_files(year, data_dir)

    return create_shot_dataframe(year, data_dir)



def process_season(year, data_dir):
    """
    Process the shot data for a given season and calculate the shot-rate
    differences for each team in each spatial bin.

    Returns a DataFrame with one row per team and bin, containing the difference
    between the team's shot rate and the league-average shot rate, along with
    the bin's center coordinates.

    Args:
        year: The starting year of the season.
        data_dir: Directory where the raw JSON files are stored.
    """

    shots = create_shot_df(year, data_dir)

    shots_normalized = normalize_shot_coordinates(shots)

    # Filter for offensive shots (x >= 25) to focus on the offensive zone
    shots_offensive = shots_normalized.loc[
        shots_normalized["x_normalized"] >= 25
    ].copy()

    team_bin_counts = calculate_team_bin_counts(
        shots_offensive, x_edges, y_edges
    )

    games_per_team = count_games_per_team(shots, data_dir)

    team_shot_rates = calculate_team_shot_rates(
        team_bin_counts, games_per_team
    )

    league_shot_rates = calculate_league_shot_rates(
        shots_offensive,
        games_per_team,
        x_edges,
        y_edges
    )

    return calculate_team_league_differences(
        team_shot_rates, league_shot_rates
    )



def create_interactive_shot_map(
    team_differences,
    season,
    rink_path,
    output_path
):
    """
    Create an interactive Plotly heatmap showing the shot-rate differences
    for each team in a given season. The heatmap is overlaid on an image of
    the rink.

    Returns a Plotly figure object and saves it as an HTML file.

    Args:
        team_differences: DataFrame containing shot-rate differences for each
        team and spatial bin.
        season: Season label.
        rink_path: Path to the rink image file.
        output_path: Path where the interactive HTML file will be saved.
    """

    # Get the teams in alphabetical order
    teams = sorted(team_differences["team"].unique())

    # Use the same color scale limits for every team in this season
    max_abs = team_differences["shot_rate_difference"].abs().max()

    if pd.isna(max_abs) or max_abs == 0:
        max_abs = 0.01

    # Compute the rink image path relative to the HTML file
    rink_url = os.path.relpath(
        Path(rink_path).resolve(),
        start=output_path.parent.resolve()
    ).replace(os.sep, "/")

    # Create the Plotly figure
    fig = go.Figure()

    # Add one heatmap trace for each team
    for team in teams:

        team_data = team_differences[
            team_differences["team"] == team
        ].copy()

        # Arrange the values into a 2D grid
        heatmap_data = (
            team_data
            .pivot(
                index="y_bin",
                columns="x_bin",
                values="shot_rate_difference"
            )
            .sort_index()
            .sort_index(axis=1)
        )

        # Get the bin center coordinates
        x_values = (
            team_data[["x_bin", "x_center"]]
            .drop_duplicates()
            .sort_values("x_bin")["x_center"]
            .tolist()
        )

        y_values = (
            team_data[["y_bin", "y_center"]]
            .drop_duplicates()
            .sort_values("y_bin")["y_center"]
            .tolist()
        )

        # Add the heatmap trace for this team
        fig.add_trace(
            go.Heatmap(
                x=x_values,
                y=y_values,
                z=heatmap_data.values,
                name=team,
                coloraxis="coloraxis",
                opacity=0.85,
                visible=(team == teams[0]),
                hovertemplate=(
                    "x: %{x}<br>"
                    "y: %{y}<br>"
                    "Difference: %{z:.4f}"
                    "<extra></extra>"
                )
            )
        )

    # Add the rink image behind the heatmap
    fig.add_layout_image(
        dict(
            source=rink_url,
            xref="x",
            yref="y",
            x=-100,
            y=42.5,
            sizex=200,
            sizey=85,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Create the team dropdown list for the interactive menu
    buttons = []

    for i, team in enumerate(teams):

        # Set visibility for the selected team and hide others
        visibility = [False] * len(teams)
        visibility[i] = True

        # Add a button for this team to the dropdown menu
        buttons.append(
            dict(
                label=team,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title.text": (
                            f"NHL 5v5 Offence : Shot-rate difference — Season {season} — {team}"
                        )
                    }
                ]
            )
        )

    # Configure the layout
    fig.update_layout(
        title=dict(
            text=f"NHL 5v5 Offence : Shot-rate difference — Season {season} — {teams[0]}",
            x=0.22,
            xanchor="left"
        ),
        coloraxis=dict(
            colorscale="RdBu_r",
            cmin=-max_abs,
            cmax=max_abs,
            cmid=0,
            colorbar=dict(
                title="Difference in <br>shots per <br>team-hour",
                x=0.92,
                xanchor="left",
                y=0.5,
                len=0.85,
                thickness=18
            )
        ),
        xaxis=dict(
            title="Distance from goal line (ft)",
            range=[0, 100],
            autorange=False,
            dtick=25,
            constrain="domain"
        ),
        yaxis=dict(
            title = "Distance from center of rink (ft)",
            range=[-42.5, 42.5],
            dtick=10,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain"
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.90,
                y=1.15,
                xanchor="right",
                yanchor="top"
            )
        ],
        width=1100,
        height=600,
        margin=dict(l=60, r=220, t=130, b=60)
    )

    # Export the figure as an HTML file
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True
    )

    print(f"Saved: {output_path.resolve()}")

    return fig



# Years to process
years = range(2016, 2021)

for year in years:

    # Season label for file naming (example : "2016-17")
    season_label = f"{year}-{str(year + 1)[-2:]}"

    # Calculate the shot-rate differences for this season
    team_differences = process_season(year, DATA_DIR)

    # Define the output HTML path
    output_path = output_dir / f"shot_map_{season_label}.html"

    # Create and save the interactive Plotly figure
    create_interactive_shot_map(
        team_differences=team_differences,
        season=season_label,
        rink_path=rink_path,
        output_path=output_path
    )