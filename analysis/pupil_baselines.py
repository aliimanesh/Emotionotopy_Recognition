"""Evaluation-protocol checks using pupil features and Emotionotopy labels.

Answers three questions without needing the PPG file or a GPU:
  1. What is the real chance level after thresholding the labels at 0.5?
  2. How well can a model do under the random window split when it only knows
     *which movie moment* a window comes from (no physiology at all)?
  3. How do pupil features do when whole movie runs are held out
     (leave-one-run-out), compared with a circular-shift null?

Usage:
    python analysis/pupil_baselines.py PUPIL_CSV LABELS_CSV [--out results.csv]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

TARGETS = ["Polarity", "Complexity", "Intensity"]
PUPIL = ["Mean", "Median", "Variance", "Max"]


def load(pupil_csv, labels_csv):
    pupil = pd.read_csv(pupil_csv)
    labels = pd.read_csv(labels_csv)[TARGETS].iloc[: len(pupil)]
    df = pd.concat([pupil, labels], axis=1)
    # Movie time = window position within each participant's full viewing.
    df["Movie_Window"] = df.groupby("Participant_Number").cumcount()
    df = df[df["Zero_Flag"] == 1].dropna(subset=PUPIL + TARGETS)
    return df.reset_index(drop=True)


def scores(y_true, y_pred):
    out = {}
    for i, t in enumerate(TARGETS):
        yt, yp = y_true[:, i] > 0.5, y_pred[:, i] > 0.5
        out[f"{t}_acc"] = accuracy_score(yt, yp)
        out[f"{t}_bacc"] = balanced_accuracy_score(yt, yp)
        out[f"{t}_r"] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    out["macro_acc"] = np.mean([out[f"{t}_acc"] for t in TARGETS])
    out["macro_bacc"] = np.mean([out[f"{t}_bacc"] for t in TARGETS])
    return out


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def fit_predict(make_model, X_tr, y_tr, X_te):
    return np.column_stack([make_model().fit(X_tr, y_tr[:, i]).predict(X_te) for i in range(y_tr.shape[1])])


def random_split(df, features, make_model, seed=42):
    """The original protocol: random 80/20 split over all windows."""
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed)
    X, y = df[features].values, df[TARGETS].values
    return scores(y[te], fit_predict(make_model, X[tr], y[tr], X[te]))


def leave_one_run_out(df, features, make_model, labels=None):
    """Hold out one movie run (for every participant) per fold."""
    y = df[TARGETS].values if labels is None else labels
    X = df[features].values
    pred = np.zeros_like(y)
    for run in sorted(df["Run_Number"].unique()):
        te = (df["Run_Number"] == run).values
        pred[te] = fit_predict(make_model, X[~te], y[~te], X[te])
    return scores(df[TARGETS].values if labels is None else labels, pred)


def circular_shift_labels(df, rng):
    """Shift the label time course within each run by a random offset (same for all participants)."""
    y = df[TARGETS].values.copy()
    out = np.empty_like(y)
    for run in df["Run_Number"].unique():
        run_mask = (df["Run_Number"] == run).values
        windows = df.loc[run_mask, "Movie_Window"].values
        uniq = np.unique(windows)
        lut = df.loc[run_mask].groupby("Movie_Window")[TARGETS].first().reindex(uniq).values
        k = rng.integers(len(uniq) // 10, len(uniq) - len(uniq) // 10)
        shifted = np.roll(lut, k, axis=0)
        out[run_mask] = shifted[np.searchsorted(uniq, windows)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pupil_csv")
    ap.add_argument("labels_csv")
    ap.add_argument("--out", default="pupil_baseline_results.csv")
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()

    df = load(args.pupil_csv, args.labels_csv)
    print(f"{len(df)} valid windows, {df['Participant_Number'].nunique()} participants")

    y = df[TARGETS].values
    rows = []
    prev = {f"{t}_acc": max((y[:, i] > 0.5).mean(), 1 - (y[:, i] > 0.5).mean()) for i, t in enumerate(TARGETS)}
    prev.update({f"{t}_frac_above_0.5": (y[:, i] > 0.5).mean() for i, t in enumerate(TARGETS)})
    prev["macro_acc"] = np.mean([prev[f"{t}_acc"] for t in TARGETS])
    rows.append({"analysis": "majority class (chance)", **prev})

    knn = lambda: KNeighborsRegressor(n_neighbors=5)
    rows.append({"analysis": "random split | movie time only (kNN)", **random_split(df, ["Movie_Window"], knn)})
    rows.append({"analysis": "random split | pupil (GBM)", **random_split(df, PUPIL, gbm)})
    rows.append({"analysis": "leave-one-run-out | pupil (GBM)", **leave_one_run_out(df, PUPIL, gbm)})

    rng = np.random.default_rng(0)
    null = [leave_one_run_out(df, PUPIL, gbm, labels=circular_shift_labels(df, rng)) for _ in range(args.n_perm)]
    null_df = pd.DataFrame(null)
    rows.append({"analysis": f"leave-one-run-out | pupil, circular-shift null mean (n={args.n_perm})", **null_df.mean().to_dict()})
    rows.append({"analysis": f"leave-one-run-out | pupil, circular-shift null 95th pct", **null_df.quantile(0.95).to_dict()})

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    cols = ["analysis", "macro_acc", "macro_bacc"] + [f"{t}_{m}" for t in TARGETS for m in ("acc", "bacc", "r")]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(res[[c for c in cols if c in res]].to_string(index=False))
        print(res[["analysis"] + [f"{t}_frac_above_0.5" for t in TARGETS]].head(1).to_string(index=False))


if __name__ == "__main__":
    main()
