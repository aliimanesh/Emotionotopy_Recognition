"""Leave-one-run-out (LORO) evaluation of the CNN / CNN-LSTM models.

Each fold holds out one movie run for *all* participants, so no movie moment
appears in both training and test. Inside the training runs, one further run
is held out for early stopping. Results for every fold and seed are appended
to a CSV so an interrupted Colab session can resume.

Colab example:
    from google.colab import drive; drive.mount('/content/drive')
    !python loro_deep_models.py --data-dir "/content/drive/My Drive/Colab Notebooks" \
        --arch cnn_lstm --inputs ppg+pupil --targets Polarity Complexity Intensity --seeds 0 1 2

Single-task control:   --targets Complexity
Two-target ablation:   --targets Polarity Intensity
PPG only:              --inputs ppg
Circular-shift null:   --null-shifts 5   (adds runs with time-shifted labels)

Test-set predictions are saved next to the results CSV (<out>_predictions/)
for analysis/loro_stats.py, which computes confidence intervals and p-values.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

PUPIL = ["Mean", "Median", "Variance", "Max"]


def load(data_dir, labels_file):
    ppg = pd.read_csv(os.path.join(data_dir, "downsampled_preprocessed_flagged_PPG_diminished.csv"))
    pupil = pd.read_csv(os.path.join(data_dir, "Pupil_Features_diminished.csv"))
    labels = pd.read_csv(os.path.join(data_dir, labels_file)).iloc[: len(ppg)]
    keep = ppg.index[ppg["Data_Origin"] == 1].intersection(pupil.index[pupil["Zero_Flag"] == 1])
    keep = keep[~labels.loc[keep, ["Polarity", "Complexity", "Intensity"]].isna().any(axis=1).values]
    movie_window = ppg.groupby("Participant_Number").cumcount()
    return {
        "ppg": ppg.loc[keep, "Data_1":"Data_200"].values[..., np.newaxis].astype("float32"),
        "pupil": pupil.loc[keep, PUPIL].values.astype("float32"),
        "labels": labels.loc[keep],
        "run": ppg.loc[keep, "Run_Number"].values,
        "window": movie_window.loc[keep].values,
        "participant": ppg.loc[keep, "Participant_Number"].values,
    }


def build(arch, inputs, n_out, heads):
    ins, feats = [], []
    if "ppg" in inputs:
        x_in = layers.Input(shape=(200, 1))
        x = layers.Conv1D(512, 50, activation="relu")(x_in)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv1D(256, 25, activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        if arch == "cnn":
            x = layers.Conv1D(128, 20, activation="relu")(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.LSTM(256, return_sequences=True)(x)
            x = layers.LSTM(128)(x)
        ins.append(x_in)
        feats.append(x)
    if "pupil" in inputs:
        p_in = layers.Input(shape=(len(PUPIL),))
        ins.append(p_in)
        feats.append(p_in)
    h0 = layers.concatenate(feats) if len(feats) > 1 else feats[0]
    outs = []
    for _ in range(heads):
        h = h0
        for units, bn, drop in [(729, True, True), (486, True, False), (324, False, False),
                                (216, False, True), (144, True, False), (96, False, True)]:
            h = layers.Dense(units)(h)
            if bn:
                h = layers.BatchNormalization()(h)
            h = layers.Activation("relu")(h)
            if drop:
                h = layers.Dropout(0.5)(h)
        outs.append(layers.Dense(n_out)(h))
    out = layers.average(outs) if heads > 1 else outs[0]
    model = models.Model(ins, out)
    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model


def circular_shift(y, run, window, rng):
    """Shift each run's label time course by a random offset (identical for all participants)."""
    out = np.empty_like(y)
    for r in np.unique(run):
        m = run == r
        uniq = np.unique(window[m])
        lut = pd.DataFrame(y[m]).groupby(window[m]).first().reindex(uniq).values
        k = rng.integers(len(uniq) // 10, len(uniq) - len(uniq) // 10)
        out[m] = np.roll(lut, k, axis=0)[np.searchsorted(uniq, window[m])]
    return out


def evaluate(y, pred, targets):
    row = {}
    for i, t in enumerate(targets):
        yt, yp = y[:, i] > 0.5, pred[:, i] > 0.5
        row[f"{t}_acc"] = accuracy_score(yt, yp)
        row[f"{t}_bacc"] = balanced_accuracy_score(yt, yp)
        row[f"{t}_majority"] = max(yt.mean(), 1 - yt.mean())
        row[f"{t}_r"] = pearsonr(y[:, i], pred[:, i])[0]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-file", default="emotion_dimension_normalized.csv")
    ap.add_argument("--arch", choices=["cnn", "cnn_lstm"], default="cnn_lstm")
    ap.add_argument("--inputs", choices=["ppg+pupil", "ppg", "pupil"], default="ppg+pupil")
    ap.add_argument("--targets", nargs="+", default=["Polarity", "Complexity", "Intensity"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--null-shifts", type=int, default=0, help="extra runs with circularly shifted labels (seed 0)")
    ap.add_argument("--out", default=None, help="results CSV (default: one file per configuration)")
    args = ap.parse_args()

    d = load(args.data_dir, args.labels_file)
    y_true = d["labels"][args.targets].values.astype("float32")
    runs = np.unique(d["run"])
    config = f"{args.arch}|{args.inputs}|{'+'.join(args.targets)}"
    if args.out is None:
        args.out = f"loro_{args.arch}_{args.inputs}_{'-'.join(args.targets)}.csv".replace("+", "_")
    pred_dir = os.path.splitext(args.out)[0] + "_predictions"
    os.makedirs(pred_dir, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["config"], prev["label_set"], prev["seed"], prev["test_run"]))

    jobs = [("real", s, y_true) for s in args.seeds]
    rng = np.random.default_rng(123)
    jobs += [(f"null{k}", 0, circular_shift(y_true, d["run"], d["window"], rng)) for k in range(args.null_shifts)]

    for label_set, seed, y in jobs:
        for test_run in runs:
            if (config, label_set, seed, test_run) in done:
                continue
            val_run = runs[(np.where(runs == test_run)[0][0] + 1) % len(runs)]
            te, va = d["run"] == test_run, d["run"] == val_run
            tr = ~(te | va)
            X = [d[k] for k in ("ppg", "pupil") if k in args.inputs]
            tf.keras.utils.set_random_seed(seed)
            model = build(args.arch, args.inputs, len(args.targets), args.heads)
            t0 = time.time()
            hist = model.fit([x[tr] for x in X], y[tr], validation_data=([x[va] for x in X], y[va]),
                             epochs=args.max_epochs, batch_size=args.batch_size, verbose=0,
                             callbacks=[tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)])
            pred = model.predict([x[te] for x in X], verbose=0)
            fold = pd.DataFrame({"participant": d["participant"][te], "run": d["run"][te], "window": d["window"][te]})
            for i, t in enumerate(args.targets):
                fold[f"y_{t}"], fold[f"pred_{t}"] = y[te][:, i], pred[:, i]
            fold.to_csv(os.path.join(pred_dir, f"{label_set}_seed{seed}_run{test_run}.csv"), index=False)
            row = {"config": config, "label_set": label_set, "seed": seed, "test_run": test_run, "val_run": val_run,
                   "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1, "minutes": (time.time() - t0) / 60,
                   **evaluate(y[te], pred, args.targets)}
            res = pd.DataFrame([row])
            if os.path.exists(args.out):
                res = pd.concat([pd.read_csv(args.out), res], ignore_index=True)
            res.to_csv(args.out, index=False)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    res = pd.read_csv(args.out)
    res = res[res["config"] == config].dropna(axis=1, how="all")
    metric_cols = [c for c in res.columns if c.endswith(("_acc", "_bacc", "_r", "_majority"))]
    summary = res.groupby("label_set")[metric_cols].agg(["mean", "std"])
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(summary.T)


if __name__ == "__main__":
    main()
