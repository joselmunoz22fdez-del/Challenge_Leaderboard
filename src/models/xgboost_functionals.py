#!/usr/bin/env python3
"""
XGBoost sobre features estadísticas por ventana.

- Features: los 72 funcionales del baseline (mean, std, var, min, max, median
  por canal) + extra opcionales (percentiles, rango, energía y magnitud por sensor).
- Validación: Leave-One-Subject-Out sobre los sujetos de train (macro F1).
- Submission: entrena con todo train y escribe el CSV con create_submission.

Uso (desde la raíz del repo):
    python src/models/xgboost_functionals.py --data_dir data/raw
    python src/models/xgboost_functionals.py --data_dir data/raw --skip_loso
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from challenge_utils import (VALID_CLASSES, compute_functionals, create_submission,  # noqa: E402
                             load_test, load_train)


def extra_features(X: np.ndarray) -> np.ndarray:
    """Percentiles, rango y energía por canal + funcionales de la magnitud de cada sensor."""
    p10, p90 = np.percentile(X, [10, 90], axis=1)
    rng = X.max(axis=1) - X.min(axis=1)
    energy = np.mean(X ** 2, axis=1)
    mag = np.linalg.norm(X.reshape(len(X), X.shape[1], 4, 3), axis=3)  # (N, 200, 4 sensores)
    mag_f = np.hstack([mag.mean(1), mag.std(1), mag.min(1), mag.max(1)])
    return np.hstack([p10, p90, rng, energy, mag_f])


def build_features(X: np.ndarray, feature_set: str) -> np.ndarray:
    F = compute_functionals(X)
    if feature_set == "ext":
        F = np.hstack([F, extra_features(X)])
    return F.astype(np.float32)


def make_model(args, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="multi:softprob", eval_metric="mlogloss",
        n_jobs=-1, random_state=seed,
    )


def loso(F, y_idx, subjects, args):
    subs = sorted(np.unique(subjects), key=lambda s: int(s[1:]))
    oof = np.zeros((len(y_idx), len(VALID_CLASSES)), dtype=np.float32)
    per_sub = []
    for s in subs:
        tr, va = subjects != s, subjects == s
        model = make_model(args).fit(F[tr], y_idx[tr])
        oof[va] = model.predict_proba(F[va])
        f1 = f1_score(y_idx[va], oof[va].argmax(1), average="macro")
        per_sub.append(f1)
        print(f"  {s:>4s}: macro F1 {100 * f1:5.2f} %", flush=True)
    pooled = f1_score(y_idx, oof.argmax(1), average="macro")
    print(f"\n  LOSO macro F1 (global): {100 * pooled:.2f} %  |  "
          f"media por sujeto {100 * np.mean(per_sub):.2f} ± {100 * np.std(per_sub):.2f} %")
    os.makedirs("results", exist_ok=True)
    np.save("results/oof_xgboost.npy", oof)
    return pooled


def main():
    ap = argparse.ArgumentParser(description="XGBoost + features estadísticas (Tennis HAR).")
    ap.add_argument("--data_dir", default="data/raw", help="Carpeta con train.parquet y test_unlabeled.parquet")
    ap.add_argument("--features", choices=["base", "ext"], default="ext")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--skip_loso", action="store_true")
    ap.add_argument("--output", default="submissions/submission_xgboost.csv")
    args = ap.parse_args()

    X_tr, y_tr, meta_tr = load_train(args.data_dir, prefer_format="parquet")
    X_te, meta_te = load_test(args.data_dir, labeled=False, prefer_format="parquet")
    print(f"Train {X_tr.shape} | Test {X_te.shape} | features={args.features}")

    t0 = time.time()
    F_tr, F_te = build_features(X_tr, args.features), build_features(X_te, args.features)
    print(f"Features: {F_tr.shape[1]} por ventana ({time.time() - t0:.1f} s)")

    classes = np.array(VALID_CLASSES)
    y_idx = np.searchsorted(classes, np.asarray(y_tr))  # VALID_CLASSES está ordenada

    if not args.skip_loso:
        print("\nLOSO:")
        loso(F_tr, y_idx, meta_tr["subject_id"].values, args)

    model = make_model(args).fit(F_tr, y_idx)
    proba = model.predict_proba(F_te)
    os.makedirs("results", exist_ok=True)
    np.save("results/test_proba_xgboost.npy", proba)
    y_pred = classes[proba.argmax(1)]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    create_submission(meta_te["sample_id"].values, y_pred, output_file=args.output)


if __name__ == "__main__":
    main()
