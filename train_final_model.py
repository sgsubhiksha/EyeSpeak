"""
Train the final classifier on your chosen 5-word vocabulary and save it to disk.

Setup:
    pip install scikit-learn pandas joblib

Run:
    python train_final_model.py
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data"
MODEL_PATH = "final_model.joblib"

FINAL_WORDS = ["book", "carrot", "down", "help", "no"]


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


def main():
    X = []
    y = []

    for word in FINAL_WORDS:
        word_dir = os.path.join(DATA_DIR, word)
        csv_files = glob.glob(os.path.join(word_dir, "*.csv"))
        print(f"{word}: {len(csv_files)} recordings")

        for csv_path in csv_files:
            feats = extract_features(csv_path)
            if feats is not None:
                X.append(feats)
                y.append(word)

    X = np.array(X)
    y = np.array(y)

    print(f"\nTraining on {len(X)} total recordings across {len(FINAL_WORDS)} words...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()