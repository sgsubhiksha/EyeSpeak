"""
Train the classifier on blendshape-based recordings (data_v2/).

Setup:
    pip install scikit-learn pandas joblib

Run:
    python train_final_model_v2.py
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data_v2"
MODEL_PATH = "final_model_v2.joblib"

BLENDSHAPES_OF_INTEREST = [
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
]


def extract_features(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    feats = []
    for col in BLENDSHAPES_OF_INTEREST:
        series = df[col]
        feats.extend([series.mean(), series.max(), series.max() - series.min()])
    return feats


def main():
    X, y = [], []
    words = sorted(os.listdir(DATA_DIR))
    print(f"Words found: {words}")

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

    X = np.array(X)
    y = np.array(y)
    print(f"\nTraining on {len(X)} recordings, {X.shape[1]} features each...")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()