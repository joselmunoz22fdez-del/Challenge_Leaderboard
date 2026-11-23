#!/usr/bin/env python3
"""
CNN 1D sencilla sobre la señal cruda (200 muestras x 12 canales).

- Normalización: media/desviación por canal calculadas en train.
- Red: 3 bloques Conv1d + BatchNorm + ReLU + MaxPool, global average pooling y capa lineal.
- Validación: Leave-One-Subject-Out (macro F1). Submission: entrena con todo train.
- --xgb_proba: si se da el .npy de probabilidades de test de XGBoost, promedia ambos (mixto).

Uso (desde la raíz del repo):
    python src/models/cnn1d.py --data_dir data/raw
    python src/models/cnn1d.py --data_dir data/raw --skip_loso --epochs 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from challenge_utils import VALID_CLASSES, create_submission, load_test, load_train  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):
    def __init__(self, n_ch=12, n_cls=6, width=32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))

        self.body = nn.Sequential(block(n_ch, width), block(width, width * 2), block(width * 2, width * 2))
        self.head = nn.Linear(width * 2, n_cls)

    def forward(self, x):  # x: (B, 200, 12)
        return self.head(self.body(x.transpose(1, 2)).mean(-1))


def fit_predict(X_tr, y_tr, X_ev, args, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = X_tr.mean((0, 1)), X_tr.std((0, 1)) + 1e-6
    Xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEV)
    Xe = torch.tensor((X_ev - mu) / sd, dtype=torch.float32, device=DEV)

    model = SimpleCNN(width=args.width).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(Xt), device=DEV).split(args.batch):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(b).softmax(1) for b in Xe.split(1024)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="CNN 1D sencilla (Tennis HAR).")
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--skip_loso", action="store_true")
    ap.add_argument("--xgb_proba", default=None, help="Probabilidades de test de XGBoost (.npy) para promediar")
    ap.add_argument("--output", default="submissions/submission_cnn1d.csv")
    args = ap.parse_args()

    X_tr, y_tr, meta_tr = load_train(args.data_dir, prefer_format="parquet")
    X_te, meta_te = load_test(args.data_dir, labeled=False, prefer_format="parquet")
    X_tr, X_te = X_tr.astype(np.float32), X_te.astype(np.float32)
    classes = np.array(VALID_CLASSES)
    y_idx = np.searchsorted(classes, np.asarray(y_tr))
    print(f"Train {X_tr.shape} | Test {X_te.shape} | device={DEV}")

    if not args.skip_loso:
        subjects = meta_tr["subject_id"].values
        subs = sorted(np.unique(subjects), key=lambda s: int(s[1:]))
        oof = np.zeros((len(X_tr), len(classes)), dtype=np.float32)
        per_sub = []
        for s in subs:
            t0 = time.time()
            tr, va = subjects != s, subjects == s
            oof[va] = fit_predict(X_tr[tr], y_idx[tr], X_tr[va], args)
            f1 = f1_score(y_idx[va], oof[va].argmax(1), average="macro")
            per_sub.append(f1)
            print(f"  {s:>4s}: macro F1 {100 * f1:5.2f} %  ({time.time() - t0:.0f} s)", flush=True)
        pooled = f1_score(y_idx, oof.argmax(1), average="macro")
        print(f"\n  LOSO macro F1 (global): {100 * pooled:.2f} %  |  "
              f"media por sujeto {100 * np.mean(per_sub):.2f} ± {100 * np.std(per_sub):.2f} %")
        os.makedirs("results", exist_ok=True)
        np.save("results/oof_cnn1d.npy", oof)

    proba = fit_predict(X_tr, y_idx, X_te, args)
    os.makedirs("results", exist_ok=True)
    np.save("results/test_proba_cnn1d.npy", proba)
    if args.xgb_proba:
        proba = (proba + np.load(args.xgb_proba)) / 2
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    create_submission(meta_te["sample_id"].values, classes[proba.argmax(1)], output_file=args.output)


if __name__ == "__main__":
    main()
