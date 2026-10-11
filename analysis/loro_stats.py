"""Uncertainty analysis for leave-one-run-out predictions.

Independent units are participants and movie runs, not 2-s windows (windows
are autocorrelated and every participant shares the same labels). So each
metric is first computed per (participant, run) and per seed, then:

  * seed variability   : SD of the score across training seeds
  * 95% CI             : bootstrap that resamples participants and runs
  * p vs chance        : circular-shift null; the label time course of each run
                         is shifted relative to the predictions (same shift for
                         every participant), preserving autocorrelation
  * model comparison   : paired difference on the same participants/runs/seeds,
                         bootstrap CI + Wilcoxon signed-rank across participants
  * Holm correction across targets

Usage:
    python loro_stats.py PRED_DIR                 # one model vs chance
    python loro_stats.py PRED_DIR_B PRED_DIR_A    # also model B minus model A
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score

N_BOOT, N_NULL = 2000, 1000
# Units where a metric is undefined (e.g. only one class in a run) are NaN and skipped.
warnings.filterwarnings("ignore", "Mean of empty slice")


def load_preds(pred_dir):
    frames = []
    for f in glob.glob(os.path.join(pred_dir, "real_seed*_run*.csv")):
        seed = int(re.search(r"seed(\d+)", f).group(1))
        frames.append(pd.read_csv(f).assign(seed=seed))
    if not frames:
        raise SystemExit(f"no real_seed*_run*.csv files in {pred_dir}")
    df = pd.concat(frames, ignore_index=True).sort_values(["seed", "run", "participant", "window"])
    targets = [c[2:] for c in df.columns if c.startswith("y_")]
    return df, targets


def unit_scores(df, target, label_col=None):
    """Score per (seed, run, participant). Returns {metric: array [seed, run, participant]}."""
    label_col = label_col or f"y_{target}"
    seeds, runs, parts = (np.sort(df[c].unique()) for c in ("seed", "run", "participant"))
    r = np.full((len(seeds), len(runs), len(parts)), np.nan)
    bacc = np.full_like(r, np.nan)
    for (s, ru, p), g in df.groupby(["seed", "run", "participant"]):
        i, j, k = np.searchsorted(seeds, s), np.searchsorted(runs, ru), np.searchsorted(parts, p)
        y, yp = g[label_col].values, g[f"pred_{target}"].values
        if len(g) > 2 and y.std() > 0 and yp.std() > 0:
            r[i, j, k] = np.corrcoef(y, yp)[0, 1]
        if 0 < (y > 0.5).mean() < 1:
            bacc[i, j, k] = balanced_accuracy_score(y > 0.5, yp > 0.5)
    return {"r": r, "bacc": bacc}


def bootstrap_ci(S, rng):
    """S: [seed, run, participant]. Resample runs and participants with replacement."""
    n_run, n_part = S.shape[1:]
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        rr, pp = rng.integers(0, n_run, n_run), rng.integers(0, n_part, n_part)
        stats[b] = np.nanmean(S[:, rr][:, :, pp])
    return np.nanpercentile(stats, [2.5, 97.5])


def shift_labels(df, target, rng):
    """Circularly shift each run's label time course (same shift for all participants and seeds)."""
    y = df[f"y_{target}"].values.copy()
    for ru in df["run"].unique():
        m = (df["run"] == ru).values
        windows = df.loc[m, "window"].values
        uniq = np.unique(windows)
        lut = df.loc[m].groupby("window")[f"y_{target}"].first().reindex(uniq).values
        k = rng.integers(len(uniq) // 10, len(uniq) - len(uniq) // 10)
        y[m] = np.roll(lut, k)[np.searchsorted(uniq, windows)]
    return y


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def summarise_model(df, targets, rng, n_null):
    rows = []
    for t in targets:
        sc = unit_scores(df, t)
        null = np.empty((n_null, 2))
        for n in range(n_null):
            shifted = df.assign(_y=shift_labels(df, t, rng))
            ns = unit_scores(shifted, t, label_col="_y")
            null[n] = [np.nanmean(ns["r"]), np.nanmean(ns["bacc"])]
        for col, metric in enumerate(("r", "bacc")):
            S = sc[metric]
            obs = np.nanmean(S)
            lo, hi = bootstrap_ci(S, rng)
            rows.append({"target": t, "metric": metric, "mean": obs, "ci_low": lo, "ci_high": hi,
                         "seed_sd": np.nanstd(np.nanmean(S, axis=(1, 2))),
                         "null_mean": null[:, col].mean(), "null_95pct": np.percentile(null[:, col], 95),
                         "p_vs_null": (1 + (null[:, col] >= obs).sum()) / (1 + n_null)})
    out = pd.DataFrame(rows)
    for metric in ("r", "bacc"):
        m = out["metric"] == metric
        out.loc[m, "p_holm"] = holm(out.loc[m, "p_vs_null"])
    return out


def compare(df_b, df_a, targets, rng):
    rows = []
    for t in targets:
        sb, sa = unit_scores(df_b, t), unit_scores(df_a, t)
        for metric in ("r", "bacc"):
            n_seed = min(sb[metric].shape[0], sa[metric].shape[0])
            D = sb[metric][:n_seed] - sa[metric][:n_seed]
            lo, hi = bootstrap_ci(D, rng)
            per_participant = np.nanmean(D, axis=(0, 1))
            p = wilcoxon(per_participant[~np.isnan(per_participant)]).pvalue
            rows.append({"target": t, "metric": metric, "diff_B_minus_A": np.nanmean(D),
                         "ci_low": lo, "ci_high": hi, "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    for metric in ("r", "bacc"):
        m = out["metric"] == metric
        out.loc[m, "p_holm"] = holm(out.loc[m, "wilcoxon_p"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir_b")
    ap.add_argument("pred_dir_a", nargs="?")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    fmt = {"display.width": 200, "display.max_columns": 20, "display.float_format": "{:.4f}".format}

    df_b, targets_b = load_preds(args.pred_dir_b)
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print(f"== {args.pred_dir_b} vs circular-shift null")
        print(summarise_model(df_b, targets_b, rng, args.n_null).to_string(index=False))
        if args.pred_dir_a:
            df_a, targets_a = load_preds(args.pred_dir_a)
            shared = [t for t in targets_b if t in targets_a]
            print(f"\n== {args.pred_dir_b} minus {args.pred_dir_a} (paired; targets {shared})")
            print(compare(df_b, df_a, shared, rng).to_string(index=False))


if __name__ == "__main__":
    main()
