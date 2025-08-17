#!/usr/bin/env python3
"""
train_model.py — Train the dual-input (metrics + code embedding) multi-label smell detector
and save BOTH the model weights and the fitted StandardScaler so inference uses
identical preprocessing.

Robust to:
- Labels CSV **with or without** an `id` column (aligns by id or by row order).
- Labels embedded inside the metrics CSV (auto-detects **numeric** label columns only).
- Arbitrary embeddings formats (.pkl/.pt/.npy/.npz; dict- or row-aligned).

Tip: To avoid accidental string columns being treated as labels, you may set
`label_cols` explicitly in config.json. Otherwise the script will auto-pick
numeric columns not in `metric_cols`.

Outputs:
  - models/anti_pattern_model_{project}.pth          (PyTorch state_dict)
  - data/scaler_{project}.npz                        (StandardScaler mean_/scale_)
  - results/{project}_training_curve.png             (loss curve)

Usage examples:
  python train_model.py --config config.json --epochs 30 --batch_size 64 --lr 1e-3

Optional overrides:
  --metrics_csv path/to/metrics.csv
  --embeddings  path/to/embeddings.pkl|.npy|.pt|.npz
  --labels_csv  path/to/labels.csv   (if omitted and labels are in metrics.csv, autodetect)
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------- Config ----------------------------------

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # sane defaults
    cfg.setdefault("models_path", "models")
    cfg.setdefault("results_path", "results")
    cfg.setdefault("data_path", "data")
    cfg.setdefault("metrics_path", "data/metrics")
    cfg.setdefault("embeddings_path", "embeddings")
    cfg.setdefault("project", "project")
    cfg.setdefault("num_smells", 11)
    cfg.setdefault("label_cols", [])  # optional explicit list
    if "metric_cols" not in cfg:
        raise ValueError("config.json must define 'metric_cols' in the order used for training.")
    return cfg

# ---------------------------- Model -----------------------------------

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x) if x.size(0) > 1 else x

class DualNN(nn.Module):
    """Dual-input network: numeric metrics + code embedding -> multi-label smells."""
    def __init__(self, metrics_dim: int, embed_dim: int, num_smells: int):
        super().__init__()
        self.m_branch = nn.Sequential(
            nn.Linear(metrics_dim, 128),
            ConditionalBatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.c_branch = nn.Sequential(
            nn.Linear(embed_dim, 256),
            ConditionalBatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_smells),
        )
    def forward(self, metrics_x: torch.Tensor, code_x: torch.Tensor) -> torch.Tensor:
        m = self.m_branch(metrics_x)
        c = self.c_branch(code_x)
        x = torch.cat([m, c], dim=1)
        return self.head(x)

# ------------------------- Data Utilities ------------------------------

def safe_mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

class SmellDataset(Dataset):
    def __init__(self, Xm: np.ndarray, Xc: np.ndarray, Y: np.ndarray):
        assert Xm.shape[0] == Xc.shape[0] == Y.shape[0]
        self.Xm = Xm.astype(np.float32)
        self.Xc = Xc.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, idx):
        return self.Xm[idx], self.Xc[idx], self.Y[idx]

# -------------------- Loading metrics/embeddings/labels ----------------

def load_embeddings(path: str, ids: List[str] | None, expected_rows: int | None) -> np.ndarray:
    """Load embeddings as either dict[id]->vec (.pkl/.pt/.npz) or row-aligned (.npy/.npz/.pt).
    If ids is None, expects row-aligned arrays of length expected_rows.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    def as_row_aligned(arr: np.ndarray) -> np.ndarray:
        if expected_rows is not None and arr.shape[0] != expected_rows:
            raise ValueError(f"Embeddings rows ({arr.shape[0]}) != expected ({expected_rows})")
        return arr.astype(np.float32)

    if p.suffix == ".npy":
        return as_row_aligned(np.load(p))

    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        if "embeddings" in d:
            return as_row_aligned(d["embeddings"]) 
        if "dict" in d and ids is not None:
            emb_dict = d["dict"].item()
            dim = len(next(iter(emb_dict.values()))) if emb_dict else 768
            out = np.zeros((len(ids), dim), dtype=np.float32)
            miss = 0
            for i, k in enumerate(ids):
                v = emb_dict.get(k)
                if v is None:
                    miss += 1
                else:
                    out[i] = np.asarray(v, dtype=np.float32)
            if miss:
                print(f"[warn] missing embeddings for {miss} ids; filled zeros")
            return out
        raise ValueError("Unsupported .npz structure for embeddings")

    if p.suffix == ".pt":
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return as_row_aligned(obj.numpy())
        elif isinstance(obj, dict) and ids is not None:
            dim = len(next(iter(obj.values()))) if obj else 768
            out = np.zeros((len(ids), dim), dtype=np.float32)
            miss = 0
            for i, k in enumerate(ids):
                v = obj.get(k)
                if v is None:
                    miss += 1
                else:
                    out[i] = np.asarray(v, dtype=np.float32)
            if miss:
                print(f"[warn] missing embeddings for {miss} ids; filled zeros")
            return out
        else:
            raise ValueError("Unsupported .pt embeddings content (expect tensor or dict)")

    if p.suffix == ".pkl":
        import pickle
        with open(p, "rb") as f:
            obj = pickle.load(f)
        # Accept: dict[id]->vec, ndarray (row-aligned), pandas.DataFrame, list/tuple of vectors
        import numpy as _np
        try:
            import pandas as _pd
        except Exception:
            _pd = None
        if isinstance(obj, dict) and ids is not None:
            # dict of id->vector (vector may be list/np/torch)
            # infer dim from first non-empty
            first = None
            for v in obj.values():
                if v is not None:
                    first = v
                    break
            dim = len(_np.asarray(first)) if first is not None else 768
            out = _np.zeros((len(ids), dim), dtype=_np.float32)
            miss = 0
            for i, k in enumerate(ids):
                v = obj.get(k)
                if v is None:
                    miss += 1
                else:
                    out[i] = _np.asarray(v, dtype=_np.float32)
            if miss:
                print(f"[warn] missing embeddings for {miss} ids; filled zeros")
            return out
        if isinstance(obj, _np.ndarray):
            return as_row_aligned(obj)
        if _pd is not None and isinstance(obj, _pd.DataFrame):
            return as_row_aligned(obj.values)
        if isinstance(obj, (list, tuple)):
            arr = _np.asarray(obj)
            return as_row_aligned(arr)
        # torch tensor
        if 'torch' in sys.modules and isinstance(obj, torch.Tensor):
            return as_row_aligned(obj.detach().cpu().numpy())
        raise ValueError("Unsupported .pkl embeddings content (expect dict, ndarray, DataFrame, list/tuple, or Tensor)")

    raise ValueError(f"Unsupported embeddings format: {p.suffix}")

# ------------------------- Labels Column Logic -------------------------

def pick_label_columns(df: pd.DataFrame, metric_cols: List[str], explicit: List[str], num_smells_cfg: int) -> List[str]:
    """Pick label columns.
    Priority:
      1) If explicit provided in config, use exactly those (must exist).
      2) Otherwise, accept columns that are *binary* labels: numeric with unique values subset of {0,1},
         and not in metric_cols and not 'id'.
    """
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"label_cols specified in config but missing in dataframe: {missing}")
        return explicit

    candidates = [c for c in df.columns if c not in set(metric_cols + ["id"]) ]
    binary_cols: List[str] = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors='coerce')
        uniq = set(pd.unique(s.dropna()))
        if uniq.issubset({0.0, 1.0}):
            binary_cols.append(c)
    if not binary_cols:
        raise ValueError("Could not auto-detect binary label columns. Provide --labels_csv or set 'label_cols' in config.json.")
    if len(binary_cols) != num_smells_cfg:
        print(f"[warn] Detected {len(binary_cols)} binary label columns; config num_smells={num_smells_cfg}. Proceeding with detected columns.")
    return binary_cols

# ------------------------- Scaler Save/Load ----------------------------

def save_scaler_npz(mean_: np.ndarray, scale_: np.ndarray, out_path: str):
    np.savez(out_path, mean_=mean_.astype(np.float32), scale_=scale_.astype(np.float32))

# --------------------------- Training ---------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
) -> List[float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xm, Xc, Y in loader:
            Xm = Xm.to(device)
            Xc = Xc.to(device)
            Y  = Y.to(device)
            optimizer.zero_grad()
            logits = model(Xm, Xc)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            running += loss.item() * Y.size(0)
        avg = running / len(loader.dataset)
        losses.append(avg)
        if ep % 1 == 0:
            print(f"epoch {ep:03d}/{epochs} | loss {avg:.4f}")

        if val_loader is not None and ep % max(1, epochs // 10) == 0:
            model.eval()
            v_run = 0.0
            with torch.no_grad():
                for Xm, Xc, Y in val_loader:
                    Xm, Xc, Y = Xm.to(device), Xc.to(device), Y.to(device)
                    v_run += nn.functional.binary_cross_entropy_with_logits(
                        model(Xm, Xc), Y, reduction='sum').item()
            v_avg = v_run / len(val_loader.dataset)
            print(f"  val loss: {v_avg:.4f}")

    return losses

# ------------------------------ Main ----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train dual-input smell detector and save scaler (robust labels handling).")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--metrics_csv", default=None, help="Override metrics CSV path")
    parser.add_argument("--embeddings", default=None, help="Override embeddings file path")
    parser.add_argument("--labels_csv", default=None, help="Override labels CSV path (optional if labels in metrics)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project = cfg["project"]
    metric_cols = cfg["metric_cols"]
    label_cols_cfg = cfg.get("label_cols", [])
    num_smells_cfg = int(cfg.get("num_smells", 11))

    models_path = cfg["models_path"]
    results_path = cfg["results_path"]
    data_path = cfg["data_path"]
    metrics_path = cfg["metrics_path"]
    embeddings_path = cfg["embeddings_path"]

    safe_mkdir(models_path); safe_mkdir(results_path); safe_mkdir(data_path)

    # --------- Resolve file paths ---------
    metrics_csv = args.metrics_csv or os.path.join(metrics_path, f"metrics_{project}.csv")
    embeddings_file = args.embeddings or os.path.join(embeddings_path, f"embeddings_{project}.pkl")
    labels_csv = args.labels_csv or os.path.join(data_path, "labels", f"labels_{project}.csv")

    if not Path(metrics_csv).exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    if not Path(embeddings_file).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    # --------- Load metrics first ---------
    mdf_raw = pd.read_csv(metrics_csv)
    if "id" in mdf_raw.columns:
        m_ids = mdf_raw["id"].astype(str).tolist()
    else:
        # fabricate sequential ids if not provided
        m_ids = [str(i) for i in range(len(mdf_raw))]
        mdf_raw.insert(0, "id", m_ids)
        print("[warn] metrics CSV missing 'id'; created sequential ids 0..N-1")

    # ensure metric columns exist and in order (accept common aliases like *_mean/_avg)
    alias_patterns = ["{m}_mean", "{m}_avg", "avg_{m}", "mean_{m}"]
    for m in metric_cols:
        if m not in mdf_raw.columns:
            alias_found = None
            for pat in alias_patterns:
                cand = pat.format(m=m)
                if cand in mdf_raw.columns:
                    alias_found = cand
                    break
            if alias_found is not None:
                mdf_raw[m] = pd.to_numeric(mdf_raw[alias_found], errors='coerce').fillna(0.0)
                print(f"[info] using alias column '{alias_found}' for metric '{m}'")
            else:
                print(f"[warn] metric column missing in metrics CSV: {m}; filling with zeros")
                mdf_raw[m] = 0.0
        else:
            mdf_raw[m] = pd.to_numeric(mdf_raw[m], errors='coerce').fillna(0.0)
    mdf = mdf_raw.set_index("id")

    # --------- Load labels (robust) ---------
    if Path(labels_csv).exists():
        ldf = pd.read_csv(labels_csv)
        if "id" in ldf.columns:
            labels_df = ldf.set_index("id")
            smell_cols = pick_label_columns(labels_df.reset_index(), metric_cols, label_cols_cfg, num_smells_cfg)
            ids = [i for i in m_ids if i in labels_df.index]
            if len(ids) != len(m_ids):
                print(f"[warn] {len(m_ids)-len(ids)} metric ids are missing labels; those rows will be dropped")
                mdf = mdf.loc[ids]
            Y = labels_df.loc[ids, smell_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        else:
            # no id in labels; align by row order with metrics
            if len(ldf) != len(mdf_raw):
                raise ValueError("labels rows do not match metrics rows and no 'id' to align by")
            smell_cols = pick_label_columns(ldf, metric_cols, label_cols_cfg, num_smells_cfg)
            Y = ldf[smell_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
            ids = m_ids
            print("[info] labels CSV has no 'id'; aligned by row order with metrics")
    else:
        # Allow labels embedded in metrics CSV (all numeric columns not in metric_cols + not 'id')
        smell_cols = pick_label_columns(mdf_raw, metric_cols, label_cols_cfg, num_smells_cfg)
        Y = mdf_raw[smell_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        ids = m_ids
        print(f"[info] using numeric labels embedded in metrics CSV: {smell_cols}")

    # Sanity: check for NaNs after coercion
    if np.isnan(Y).any():
        nan_cols = [c for c in smell_cols if pd.to_numeric((mdf_raw if not Path(labels_csv).exists() else pd.read_csv(labels_csv))[c], errors='coerce').isna().any()]
        raise ValueError(f"Non-numeric values found in label columns: {nan_cols}. Clean your labels or set explicit label_cols in config.json")

    # --------- Extract metrics aligned to ids ---------
    Xm = mdf.loc[ids, metric_cols].values.astype(np.float32)

    # --------- Load embeddings and align ---------
    Xc = load_embeddings(embeddings_file, ids=ids, expected_rows=len(ids))
    embed_dim = Xc.shape[1]

    # --------- Train/val split ---------
    n = len(ids)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int((1.0 - args.val_split) * n)
    tr_idx = idx[:split]
    va_idx = idx[split:]

    Xm_tr, Xm_va = Xm[tr_idx], Xm[va_idx]
    Xc_tr, Xc_va = Xc[tr_idx], Xc[va_idx]
    Y_tr,  Y_va  = Y[tr_idx],  Y[va_idx]

    # --------- Fit StandardScaler on TRAIN METRICS ONLY & save ----------
    mean_ = Xm_tr.mean(axis=0)
    std_ = Xm_tr.std(axis=0, ddof=0)
    std_safe = np.where(std_ == 0.0, 1.0, std_)
    Xm_tr_s = (Xm_tr - mean_) / std_safe
    Xm_va_s = (Xm_va - mean_) / std_safe

    scaler_npz = os.path.join(data_path, f"scaler_{project}.npz")
    np.savez(scaler_npz, mean_=mean_.astype(np.float32), scale_=std_safe.astype(np.float32))
    print(f"Saved StandardScaler params to {scaler_npz}")

    # --------- Dataloaders ---------
    train_ds = SmellDataset(Xm_tr_s, Xc_tr, Y_tr)
    val_ds   = SmellDataset(Xm_va_s, Xc_va, Y_va)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --------- Model / Train ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNN(metrics_dim=Xm.shape[1], embed_dim=embed_dim, num_smells=Y.shape[1]).to(device)

    losses = train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    # --------- Save model weights ---------
    model_path = os.path.join(models_path, f"anti_pattern_model_{project}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    # --------- Plot & save training curve ---------
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch"); plt.ylabel("BCEWithLogitsLoss")
    plt.title(f"Training Loss — {project}")
    curve_path = os.path.join(results_path, f"{project}_training_curve.png")
    plt.savefig(curve_path, dpi=160); plt.close()
    print(f"Saved training curve to {curve_path}")

if __name__ == "__main__":
    main()
