"""
Step 5: pilot test - can a simple classifier tell your two words apart?

This loads every recording under data/<word>/*.csv, turns each recording
into a compact feature vector (summary stats of width/height/ratio over
the recording), and runs cross-validated classification to check accuracy.

Setup:
    pip install scikit-learn numpy pandas

Run:
    python pilot_classifier_test.py
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "data"


def extract_features(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    feats = []
    for col in ["width", "height", "ratio"]:
        series = df[col]
        feats.extend([
            series.mean(),
            series.std() if len(series) > 1 else 0.0,
            series.min(),
            series.max(),
            series.max() - series.min(),
        ])
    return feats


def load_dataset():
    X = []
    y = []
    words = sorted(os.listdir(DATA_DIR))
    print(f"Found word folders: {words}")

    for word in words:
        word_dir = os.path.join(DATA_DIR, word)
        if not os.path.isdir(word_dir):
            continue
        csv_files = glob.glob(os.path.join(word_dir, "*.csv"))
        print(f"  {word}: {len(csv_files)} recordings")

        for csv_path in csv_files:
            feats = extract_features(csv_path)
            if feats is not None:
                X.append(feats)
                y.append(word)

    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()

    if len(set(y)) < 2:
        print("\nNeed at least 2 different words recorded to test separability.")
        return

    print(f"\nTotal usable recordings: {len(X)}")
    print(f"Feature vector size per recording: {X.shape[1]}")

    n_splits = min(5, min(np.bincount(np.unique(y, return_inverse=True)[1])))
    if n_splits < 2:
        print("\nNot enough recordings per word for cross-validation yet. "
              "Record more reps (aim for 15-20+ per word) and try again.")
        return

    print(f"\nRunning {n_splits}-fold cross-validation...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    acc = accuracy_score(y, y_pred)
    print(f"\nCross-validated accuracy: {acc * 100:.1f}%")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    labels = sorted(set(y))
    print(labels)
    print(confusion_matrix(y, y_pred, labels=labels))
    print("\nDetailed report:")
    print(classification_report(y, y_pred))

    chance = 1.0 / len(set(y))
    print(f"Chance-level accuracy with {len(set(y))} words: {chance * 100:.1f}%")
    if acc > chance + 0.25:
        print("This looks separable enough to be worth building on.")
    elif acc > chance + 0.1:
        print("Some signal here, but it's marginal - consider more reps, "
              "more distinct words, or double-checking recording consistency.")
    else:
        print("Not much separation yet - the mouth positions may be too similar, "
              "or recordings may be inconsistent. Try more visually distinct words.")


if __name__ == "__main__":
    main()