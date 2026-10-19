# Imports
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import question_3


# Get the directory containing question_4.py
DATA_DIR = Path(__file__).resolve().parent


# Question 4.1: Produce a figure comparing shot types over all teams
# Overlay the number of goals over the number of shots

# Load the 2023-24 season
season = 2023
df_shots = question_3.create_shot_dataframe(season, DATA_DIR)

# Count shots and goals for each shot type
shot_summary = (
    df_shots.groupby("shot_type")
    .agg(
        shots=("event_type", "size"),
        goals=("event_type", lambda x: (x == "goal").sum())
    )
)

# Calculate the percentage of shots that result in goals
shot_summary["goal_percentage"] = (
    100 * shot_summary["goals"] / shot_summary["shots"]
)

# Sort by number of shots
shot_summary = shot_summary.sort_values("shots", ascending=False)

# Plot the results
fig, ax1 = plt.subplots(figsize=(12, 6))

shot_summary["shots"].plot(
    kind="bar", ax=ax1, color="steelblue", label="Shots"
)

ax1.set_xlabel("Shot type")
ax1.set_ylabel("Number of shots")
ax1.set_title("Shot Types and Goals Across All Teams (2023-24)")
ax1.tick_params(axis="x", rotation=45)

# Add goals using a second y-axis
ax2 = ax1.twinx()
ax2.plot(
    range(len(shot_summary)),
    shot_summary["goals"],
    color="red",
    marker="o",
    label="Goals"
)
ax2.set_ylabel("Number of goals")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.tight_layout()
plt.show()

# Display percentages to identify the most dangerous shot type
print(shot_summary.sort_values("goal_percentage", ascending=False))


# Question 4.2: Produce a figure showing the relationship between shot distance
# and scoring probability for each season from 2018-19 to 2020-21

# Seasons to analyze
seasons = [2018, 2019, 2020]

# Loop to create one figure for each season
for year in seasons:

    # Load the season data
    df_shots = question_3.create_shot_dataframe(year, DATA_DIR)

    # Keep goals and shots on goal
    df_shots = df_shots[
        df_shots["event_type"].isin(["goal", "shot-on-goal"])
    ].copy()

    # Convert coordinates to numeric values
    df_shots["x"] = pd.to_numeric(df_shots["x"], errors="coerce")
    df_shots["y"] = pd.to_numeric(df_shots["y"], errors="coerce")

    # Remove events with missing coordinates
    df_shots = df_shots.dropna(subset=["x", "y"])

    # Calculate distance from the nearest net
    # In the NHL coordinate system, the two nets are located at x = -89 and x = 89.
    df_shots["distance"] = np.hypot(
        89 - df_shots["x"].abs(),
        df_shots["y"]
    )

    # Group shots into 5-foot distance intervals
    bin_width = 5
    max_distance = int(
        np.ceil(df_shots["distance"].max() / bin_width) * bin_width
    )

    # Create bins for the distance intervals
    bins = np.arange(0, max_distance + bin_width, bin_width)

    # Assign each shot to an interval
    df_shots["distance_bin"] = pd.cut(
        df_shots["distance"],
        bins=bins,
        right=False
    )

    # Count shots and goals in each distance interval
    summary = (
        df_shots.groupby("distance_bin", observed=True)
        .agg(
            shots=("event_type", "size"),
            goals=("event_type", lambda x: (x == "goal").sum())
        )
        .reset_index()
    )

    # Calculate goal percentage
    summary["goal_percentage"] = (
        100 * summary["goals"] / summary["shots"]
    )

    # Calculate the midpoint of each distance interval
    summary["distance_midpoint"] = summary["distance_bin"].apply(
        lambda interval: (interval.left + interval.right) / 2
    )

    # Create the plot
    plt.figure(figsize=(9, 5))

    plt.plot(
        summary["distance_midpoint"],
        summary["goal_percentage"],
        marker="o"
    )

    plt.xlabel("Distance from the net (feet)")
    plt.ylabel("Goal percentage (%)")
    plt.title(
        f"Shot Distance and Scoring Probability "
        f"({year}-{str(year + 1)[-2:]})"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Question 4.3: Produce a figure showing goal percentage as a function
# of both distance from the net and shot type

# Define parameters for the analysis
season = 2020
bin_width = 10
max_distance = 100
min_shots = 20

# Load the data
df = question_3.create_shot_dataframe(season, DATA_DIR)

# Keep goals and shots on goal only
df = df[df["event_type"].isin(["goal", "shot-on-goal"])].copy()

# Convert coordinates to numeric and remove missing values
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["x", "y", "shot_type"])

# Calculate distance from the nearest net
# In the NHL coordinate system, the two nets are located at x = -89 and x = 89.
# Hypot calculates the distance from the net using the Pythagorean theorem.
df["distance"] = np.hypot(89 - df["x"].abs(), df["y"])
df["is_goal"] = df["event_type"].eq("goal")

# Group distances into intervals with reusable labels
bins = np.arange(0, max_distance + bin_width, bin_width)
labels = [f"{b}-{b + bin_width}" for b in bins[:-1]]

df["distance_bin"] = pd.cut(
    df["distance"],
    bins=bins,
    labels=labels,
    right=False
)

# Calculate goal percentage for each shot type and distance interval
summary = (
    df.groupby(["shot_type", "distance_bin"], observed=True)
    .agg(
        shots=("is_goal", "size"),
        goals=("is_goal", "sum")
    )
    .reset_index()
)

summary["goal_percentage"] = (
    100 * summary["goals"] / summary["shots"]
)

# Hide cells with fewer than the minimum number of shots
summary.loc[
    summary["shots"] < min_shots,
    "goal_percentage"
] = np.nan

# Create a matrix for the heatmap
pivot = summary.pivot(
    index="shot_type",
    columns="distance_bin",
    values="goal_percentage"
).reindex(columns=labels)

# Sort shot types by their overall goal percentage
overall = (
    df.groupby("shot_type")["is_goal"]
    .mean()
    .sort_values(ascending=False)
)
pivot = pivot.reindex(overall.index)

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 6))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Goal percentage (%)"},
    ax=ax
)

ax.set_xlabel("Distance from net (feet)")
ax.set_ylabel("Shot type")
ax.set_title("Goal Percentage by Shot Type and Distance (2020-21)")

# Small text at the bottom left to explain blank cells
fig.text(
    0.01, 0.01,
    f"Blank cells: fewer than {min_shots} shots.",
    fontsize=8,
    style="italic"
)

plt.tight_layout()
plt.show()

# References for the analysis and visualizations:
# Pandas documentation: https://pandas.pydata.org/docs/user_guide/groupby.html
# NumPy hypot: https://numpy.org/doc/stable/reference/generated/numpy.hypot.html
# Matplotlib two-axis plot (question 4.1): https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
# Seaborn heatmap (question 4.3): https://seaborn.pydata.org/generated/seaborn.heatmap.html
