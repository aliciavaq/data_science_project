import wandb

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from ift6758.data.milestone2_Q2 import load_dev_set, split_seed, val_fraction
import joblib



def load_data():
    df = load_dev_set("ift6758/data")
    df["angle_abs"] = df["angle"].abs()
    n_total = len(df)
    df = df.dropna(subset=["distance", "angle_abs"])
    print(f"{n_total - len(df)} rows  dropped")

    train = df[df["split"]== "train"]
    val = df[df["split"]== "val"]
    return train, val


def train_model(train, features):
    # default LogisticRegression 
    clf = LogisticRegression()
    clf.fit(train[features], train["is_goal"])
    return clf


def goal_prob(clf, df, features):
    # Probability of the goal class
    goal_col = list(clf.classes_).index(1)
    return clf.predict_proba(df[features])[:, goal_col]


def random_baseline(n, seed=split_seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=n)

def save_plot(title, xlabel, ylabel, file_name):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.savefig( Path("ift6758/plots")/ file_name)


def plot_cumulative(y, goal_probs):
    plt.figure()
    for name, p in goal_probs.items():
        sorted = y.values[np.argsort(p)[::-1]]
        cum = np.cumsum(sorted) /sorted.sum() * 100
        plt.plot(np.linspace(100, 0, len(sorted)), cum, label=name)
    save_plot("Cumulative(goals/percentile)", "Shot percentile","Cumulative percentile", "q3_cumulative_goals.png")

def plot_goalrate(y, goal_probs):
    plt.figure()
    for name, p in goal_probs.items():
        sorted = y.values[np.argsort(p)[::-1]]
        rates = []
        for split in np.array_split(sorted, 20):
            rates.append(split.mean() * 100)
        plt.plot(np.arange(97.5, 0, -5), rates, label=name)
    save_plot("Goal rate vs percentile", "Shot percentile","Goal rate", "q3_goal_rate.png")

def plot_reliability(y, goal_probs):
    fig, axes = plt.subplots()
    for name, p in goal_probs.items():
        CalibrationDisplay.from_predictions(y, p, n_bins=10, name=name, ax=axes)
    save_plot("Reliability diagram","Mean predicted probability", "fraction of goals", "q3_reliability.png")

def plot_roc(y, goal_probs):
    fig, ax = plt.subplots()
    for name, p in goal_probs.items():
        RocCurveDisplay.from_predictions(y, p, name=name, ax=ax)
    save_plot("ROC curves", "False positive rate", "True positive rate", "q3_roc.png")




def main():
    models = {"distance": ["distance"],
              "angle": ["angle_abs"],
              "distance_angle": ["distance", "angle_abs"]
              ,}

    train, val = load_data()
    goal_probs = {}
    for name, features in models.items():
        clf = train_model(train, features)
        goal_probs[name] = goal_prob(clf, val, features)
        joblib.dump(clf, Path("ift6758/plots") / f"logistic_regression_{name}.joblib")
        pred = goal_probs[name] >= 0.5
        print(name, "accuracy:", round(accuracy_score(val["is_goal"], pred), 3))

    goal_probs["random"] = random_baseline(len(val))
    plot_roc(val["is_goal"], goal_probs)
    plot_goalrate(val["is_goal"], goal_probs)
    plot_cumulative(val["is_goal"], goal_probs)
    plot_reliability(val["is_goal"], goal_probs)

if __name__ == "__main__":
    main()