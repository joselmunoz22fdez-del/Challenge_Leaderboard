#!/usr/bin/env python3
"""
CNN 1D con data augmentation sobre la señal cruda (200 muestras x 12 canales).

- Entrada: los 12 ejes sin normalizar + la magnitud de cada sensor (16 canales).
- Augmentation en cada batch: rotación 3D aleatoria pequeña por sensor, ganancia por sensor,
  ruido, apagado aleatorio de un sensor y desplazamiento circular de la ventana.
- Red: 6 capas Conv1d + BatchNorm + ReLU (2 MaxPool), pooling medio + máximo, dropout y capa lineal.
- Entrenamiento: AdamW + OneCycle, label smoothing 0.1, 15 épocas. Se promedian varias semillas.
- Validación: Leave-One-Subject-Out (macro F1). Submission: entrena con todo train.

Uso (desde la raíz del repo):
    python src/models/cnn1d_aug.py --data_dir data/raw
    python src/models/cnn1d_aug.py --data_dir data/raw --skip_loso
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from challenge_utils import VALID_CLASSES, create_submission, load_test, load_train  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


def rot_mats(n, deg):
    """n rotaciones 3D pequeñas (eje aleatorio, ángulo ~ N(0, deg)), fórmula de Rodrigues."""
    ax = F.normalize(torch.randn(n, 3, device=DEV), dim=1)
    th = torch.randn(n, 1, 1, device=DEV) * deg * np.pi / 180
    K = torch.zeros(n, 3, 3, device=DEV)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = ax[:, 2], -ax[:, 1], ax[:, 0]
    return torch.eye(3, device=DEV) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def augment(x, rot=20.0, sensor_drop=0.05):
    """x: (B, 200, 12) con los canales agrupados por sensor (x, y, z)."""
    B = x.shape[0]
    s = x.view(B, 200, 4, 3)
    R = rot_mats(B * 4, rot).view(B, 1, 4, 3, 3)
    s = (R @ s.unsqueeze(-1)).squeeze(-1)                                   # rotación por sensor
    s = s * (1 + 0.1 * torch.randn(B, 1, 4, 1, device=DEV))                 # ganancia por sensor
    s = s + 0.02 * torch.randn_like(s)                                      # ruido
    s = s * (torch.rand(B, 1, 4, 1, device=DEV) >= sensor_drop).float()     # apagar un sensor
    x = s.reshape(B, 200, 12)
    return torch.roll(x, torch.randint(0, 200, (1,)).item(), 1)             # desplazamiento circular


def add_magnitude(x):
    """(B, 200, 12) -> (B, 200, 16): añade |a| de cada sensor."""
    return torch.cat([x, x.view(x.shape[0], 200, 4, 3).norm(dim=-1)], 2)


def conv(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))


class CNN(nn.Module):
    def __init__(self, n_ch=16, n_cls=6, width=64, drop=0.3):
        super().__init__()
        w = width
        self.body = nn.Sequential(conv(n_ch, w, 7), conv(w, w, 5), nn.MaxPool1d(2),
                                  conv(w, 2 * w, 5), conv(2 * w, 2 * w, 5), nn.MaxPool1d(2),
                                  conv(2 * w, 4 * w, 3), conv(4 * w, 4 * w, 3))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(8 * w, n_cls))

    def forward(self, x):  # x: (B, 200, 12)
        z = self.body(add_magnitude(x).transpose(1, 2))
        return self.head(torch.cat([z.mean(2), z.amax(2)], 1))


def fit_predict(X_tr, y_tr, X_ev, args, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEV)
    Xe = torch.tensor(X_ev, dtype=torch.float32, device=DEV)

    model = CNN(width=args.width, drop=args.dropout).to(DEV)
    n_batches = len(Xt) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * n_batches, pct_start=0.2)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(n_batches):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            x = augment(Xt[idx], rot=args.rot)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = F.cross_entropy(model(x).float(), yt[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([F.softmax(model(b).float(), 1) for b in Xe.split(512)]).cpu().numpy()


def fit_predict_seeds(X_tr, y_tr, X_ev, args, n_seeds):
    return sum(fit_predict(X_tr, y_tr, X_ev, args, seed=s) for s in range(n_seeds)) / n_seeds


def main():
    ap = argparse.ArgumentParser(description="CNN 1D con data augmentation (Tennis HAR).")
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rot", type=float, default=20.0, help="Desviación del ángulo de rotación (grados)")
    ap.add_argument("--loso_seeds", type=int, default=3)
    ap.add_argument("--final_seeds", type=int, default=5)
    ap.add_argument("--skip_loso", action="store_true")
    ap.add_argument("--output", default="submissions/submission_cnn1d_aug.csv")
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
            oof[va] = fit_predict_seeds(X_tr[tr], y_idx[tr], X_tr[va], args, args.loso_seeds)
            f1 = f1_score(y_idx[va], oof[va].argmax(1), average="macro")
            per_sub.append(f1)
            print(f"  {s:>4s}: macro F1 {100 * f1:5.2f} %  ({time.time() - t0:.0f} s)", flush=True)
        pooled = f1_score(y_idx, oof.argmax(1), average="macro")
        print(f"\n  LOSO macro F1 (global): {100 * pooled:.2f} %  |  "
              f"media por sujeto {100 * np.mean(per_sub):.2f} ± {100 * np.std(per_sub):.2f} %")
        os.makedirs("results", exist_ok=True)
        np.save("results/oof_cnn1d_aug.npy", oof)

    proba = fit_predict_seeds(X_tr, y_idx, X_te, args, args.final_seeds)
    os.makedirs("results", exist_ok=True)
    np.save("results/test_proba_cnn1d_aug.npy", proba)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    create_submission(meta_te["sample_id"].values, classes[proba.argmax(1)], output_file=args.output)


if __name__ == "__main__":
    main()
