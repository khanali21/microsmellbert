#!/usr/bin/env python3
"""
infer_smells.py — build datasets from Java code and infer code smells with a dual-input NN.

USAGE (from folder with config.json):
  1) Prepare a new test dataset from source code:
     python infer_smells.py prepare \
       --out_name petclinic_aug16 \
       --granularity class \
       --include_tests false

  2) Run inference on an existing packed dataset (.npz):
     python infer_smells.py infer \
       --npz results/test_petclinic_aug16.npz \
       --model_path models/anti_pattern_model_spring_petclinic.pth \
       --out_prefix petclinic_aug16

  3) One-shot: prepare from code then infer:
     python infer_smells.py infer-new \
       --out_name petclinic_aug16 \
       --granularity class \
       --include_tests false \
       --model_path models/anti_pattern_model_spring_petclinic.pth

REQUIREMENTS
  - Python 3.9+
  - pip install torch transformers pandas numpy matplotlib
  - Java + CK jar (if computing metrics): https://github.com/mauricioaniche/ck

NOTES
  - Metric order MUST match the order used in training (config.json -> metric_cols).
  - If you saved a StandardScaler during training (scaler_{project}.npz in data_path),
    it will be applied at inference for correct normalization.
  - If CK doesn't provide some metrics you trained on (e.g., fanin/fanout/tcc/lcc),
    the script will fill them with zeros and warn. For best fidelity, place precomputed
    metrics CSVs in metrics_path.
"""

from __future__ import annotations

import os
import re
import io
import gc
import sys
import json
import glob
import math
import time
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# HuggingFace for code embeddings
from transformers import AutoTokenizer, AutoModel

# ---------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infer_smells")

# ---------------------------- Config ----------------------------------

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # sensible defaults
    cfg.setdefault("models_path", "models")
    cfg.setdefault("results_path", "results")
    cfg.setdefault("project", "project")
    cfg.setdefault("data_path", "data")
    cfg.setdefault("repo_path", ".")
    cfg.setdefault("services", [])  # if empty, scan whole repo_path
    cfg.setdefault("metrics_path", "./data/metrics/")
    cfg.setdefault("embeddings_path", "embeddings")
    cfg.setdefault("hf_model_name", "microsoft/codebert-base")
    cfg.setdefault("max_seq_len", 512)
    cfg.setdefault("ck_jar", "ck.jar")
    cfg.setdefault("num_smells", 11)
    cfg.setdefault("crudy_annotations", ["@GetMapping","@PostMapping","@PutMapping","@DeleteMapping"])
    cfg.setdefault("mega_loc_threshold", 500)
    cfg.setdefault("mega_classes_threshold", 10)
    if "metric_cols" not in cfg:
        raise ValueError("config.json must define 'metric_cols' in the same order used for training.")
    return cfg

# ---------------------------- Model -----------------------------------

class ConditionalBatchNorm1d(nn.Module):
    """Applies BatchNorm1d only when batch size > 1 (avoids BN instability for singletons)."""
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
        self.fused = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_smells),
            nn.Sigmoid(),
        )

    def forward(self, metrics: torch.Tensor, code_emb: torch.Tensor) -> torch.Tensor:
        m_out = self.m_branch(metrics)
        c_out = self.c_branch(code_emb)
        fused = torch.cat((m_out, c_out), dim=1)
        return self.fused(fused)

# ---------------------------- Utilities -------------------------------

def safe_mkdir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def run_ck(ck_jar: str, repo_path: str, out_path: Path, use_jar: bool = False) -> Tuple[Path, Path]:
    safe_mkdir(out_path)
    cmd = [
        "java",
        "-jar",
        ck_jar,
        repo_path,
        str(use_jar).lower(),
        "0",
        "false",
        str(out_path),
    ]
    logger.info("Running CK: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    csv_class = out_path / "class.csv"
    csv_method = out_path / "method.csv"
    if not csv_class.exists():
        raise FileNotFoundError(f"CK failed to produce class.csv in {out_path}")
    return csv_class, csv_method

def find_java_files(root: Path, include_tests: bool = False) -> List[Path]:
    files = list(root.rglob("*.java"))
    if not include_tests:
        files = [f for f in files if "test" not in str(f).lower()]
    logger.info("Java files discovered: %d", len(files))
    return files

def compute_code_embedding(
    code_str: str, tokenizer: AutoTokenizer, model: AutoModel, max_len: int = 512
) -> np.ndarray:
    inputs = tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def pack_npz(path: Path, ids: List[str], X_metrics: np.ndarray, X_code: np.ndarray, smell_names: List[str]):
    np.savez(
        path,
        ids=np.array(ids, dtype=object),
        X_metrics=X_metrics,
        X_code=X_code,
        smell_names=np.array(smell_names, dtype=object),
    )

def load_npz(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    return (
        data["ids"].tolist(),
        data["X_metrics"],
        data["X_code"],
        data["smell_names"].tolist(),
    )

# ---------------------------- Dataset Preparation -------------------------------

def prepare_dataset(
    cfg: dict,
    src_root: str | None,
    out_name: str,
    ck_jar: str,
    granularity: str = "class",
    include_tests: bool = False,
    git_diff_base: str | None = None,
) -> str:
    root = Path(src_root or cfg["repo_path"])

    # 1) Run CK for metrics
    ck_csv_class, ck_csv_method = run_ck(
        ck_jar,
        str(root),
        out_path=Path(cfg["results_path"]) / f"ck_{out_name}/",
        use_jar=False  # Force source code processing
    )

    # Load CK metrics
    metrics_df = pd.read_csv(ck_csv_class)
    metrics_df["id"] = metrics_df["file"].apply(lambda x: Path(x).stem)  # or use FQCN

    # Filter for services if specified
    if cfg["services"]:
        metrics_df = metrics_df[metrics_df["id"].isin(cfg["services"])]

    # Handle missing metrics
    missing_cols = set(cfg["metric_cols"]) - set(metrics_df.columns)
    if missing_cols:
        logger.warning("Filling missing metrics with 0: %s", missing_cols)
        for col in missing_cols:
            metrics_df[col] = 0

    # 2) Find source files
    java_files = find_java_files(root, include_tests)

    # 3) Compute custom features: crudy_count, is_mega_file
    crudy_list = []
    mega_list = []
    for f in java_files:
        code = f.read_text()
        crudy_count = sum(1 for ann in cfg["crudy_annotations"] if ann in code)
        crudy_list.append(crudy_count)
        loc = len(code.splitlines())
        is_mega = loc > cfg["mega_loc_threshold"]
        mega_list.append(int(is_mega))

    # 4) Normalize metrics if scaler available
    scaler_path = Path(cfg["data_path"]) / f"scaler_{cfg['project']}.npz"
    if scaler_path.exists():
        try:
            scaler_data = np.load(scaler_path)
            scaler_mean = scaler_data["mean"]
            scaler_std = scaler_data["std"]
            X_metrics = (metrics_df[cfg["metric_cols"]].values - scaler_mean) / scaler_std
        except (KeyError, OSError) as e:
            logger.warning(f"Failed to load or use scaler {scaler_path}: {e}. Skipping normalization.")
            X_metrics = metrics_df[cfg["metric_cols"]].values
    else:
        X_metrics = metrics_df[cfg["metric_cols"]].values

    # 5) Compute code embeddings
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model_name"])
    model = AutoModel.from_pretrained(cfg["hf_model_name"])
    logger.info("Computing code embeddings for %d files...", len(java_files))
    embs = []
    ids = []
    for f in java_files:
        code_str = f.read_text()
        emb = compute_code_embedding(code_str, tokenizer, model, cfg["max_seq_len"])
        embs.append(emb)
        ids.append(str(f.relative_to(root)))
    X_code = np.vstack(embs).astype(np.float32)

    # 6) Save human-readable CSV and packed NPZ
    smell_names = [f"smell_{i}" for i in range(int(cfg.get("num_smells", 11)))]

    out_csv = Path(cfg["results_path"]) / f"test_{out_name}.csv"
    out_npz = Path(cfg["results_path"]) / f"test_{out_name}.npz"

    dump_df = pd.DataFrame({"id": ids})
    for c in cfg["metric_cols"]:
        dump_df[c] = metrics_df[c].values
    dump_df["crudy_count"] = crudy_list
    dump_df["is_mega_file"] = mega_list
    dump_df.to_csv(out_csv, index=False)

    pack_npz(out_npz, ids, X_metrics, X_code, smell_names)
    logger.info("Prepared dataset: %s (csv), %s (npz)", out_csv, out_npz)
    return str(out_npz)

# Convenience wrapper matching the CLI wording

def prepare_new_dataset_from_code(cfg: dict, out_name: str, granularity: str = "class", include_tests: bool = False) -> str:
    return prepare_dataset(
        cfg=cfg,
        src_root=None,
        out_name=out_name,
        ck_jar=cfg.get("ck_jar", "ck.jar"),
        granularity=granularity,
        include_tests=include_tests,
        git_diff_base=None,
    )

# ---------------------------- Inference -------------------------------

def run_inference(
    cfg: dict,
    model_path: str,
    npz_path: str | None = None,
    metrics_csv: str | None = None,
    code_emb_path: str | None = None,
    out_prefix: str | None = None,
    threshold: float = 0.5,
):
    # Load data
    if npz_path:
        ids, X_metrics, X_code, smell_names = load_npz(Path(npz_path))
    else:
        # Fallback: load from CSV and embeddings
        metrics_df = pd.read_csv(metrics_csv)
        ids = metrics_df["id"].tolist()
        X_metrics = metrics_df[cfg["metric_cols"]].values
        if code_emb_path.endswith(".npy"):
            X_code = np.load(code_emb_path)
        elif code_emb_path.endswith(".pt"):
            X_code = torch.load(code_emb_path).numpy()
        else:
            raise ValueError("Unsupported code_emb_path format")
        smell_names = [f"smell_{i}" for i in range(cfg["num_smells"])]

    # Load model
    metrics_dim = X_metrics.shape[1]
    embed_dim = X_code.shape[1]
    model = DualNN(metrics_dim, embed_dim, cfg["num_smells"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_metrics_t = torch.from_numpy(X_metrics.astype(np.float32)).to(device)
    X_code_t = torch.from_numpy(X_code.astype(np.float32)).to(device)
    with torch.no_grad():
        probs = model(X_metrics_t, X_code_t).cpu().numpy()

    # Threshold to labels
    labels = (probs > threshold).astype(int)

    # Save results
    out_csv = Path(cfg["results_path"]) / f"infer_{out_prefix}.csv"
    res_df = pd.DataFrame({"id": ids})
    for i, name in enumerate(smell_names):
        res_df[name] = labels[:, i]
        res_df[f"{name}_prob"] = probs[:, i]
    res_df.to_csv(out_csv, index=False)
    logger.info("Inference results: %s", out_csv)

    # Optional: plot prob distributions
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axs.flatten()[:-1]):
        ax.hist(probs[:, i], bins=20)
        ax.set_title(smell_names[i])
    plt.tight_layout()
    plt.savefig(Path(cfg["results_path"]) / f"probs_{out_prefix}.png")
    logger.info("Probability histograms saved.")

# ------------------------------ CLI -----------------------------------

def main():
    cfg = load_config("config.json")
    safe_mkdir(cfg["results_path"])

    parser = argparse.ArgumentParser(description="Prepare test datasets from Java code and infer code smells.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare
    p_prep = sub.add_parser("prepare", help="Build test npz from fresh Java code (uses repo_path/services from config).")
    p_prep.add_argument("--out_name", required=True, help="Name suffix for test dataset.")
    p_prep.add_argument("--granularity", default="class", choices=["class", "module"], help="Aggregation level.")
    p_prep.add_argument("--include_tests", default="false", choices=["true", "false"], help="Include test sources.")

    # infer on existing dataset
    p_inf = sub.add_parser("infer", help="Run inference on an existing dataset.")
    p_inf.add_argument("--npz", default=None, help="Packed npz with ids, X_metrics, X_code, smell_names.")
    p_inf.add_argument("--metrics_csv", default=None, help="Fallback: CSV with 'id' and metric_cols.")
    p_inf.add_argument("--code_emb_path", default=None, help="Fallback: .npy or torch .pt embeddings.")
    p_inf.add_argument("--out_prefix", default=None, help="Prefix for output files.")
    p_inf.add_argument("--model_path", default=None, help="Path to model weights .pth")
    p_inf.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for labels.")

    # one-shot: prepare + infer
    p_comb = sub.add_parser("infer-new", help="Prepare from code, then infer with model.")
    p_comb.add_argument("--out_name", required=True, help="Name suffix for test dataset and outputs.")
    p_comb.add_argument("--granularity", default="class", choices=["class", "module"], help="Aggregation level.")
    p_comb.add_argument("--include_tests", default="false", choices=["true", "false"], help="Include test sources.")
    p_comb.add_argument("--model_path", default=None, help="Path to model weights .pth")
    p_comb.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for labels.")

    args = parser.parse_args()

    if args.cmd == "prepare":
        npz = prepare_new_dataset_from_code(
            cfg,
            out_name=args.out_name,
            granularity=args.granularity,
            include_tests=(args.include_tests.lower() == "true"),
        )
        logger.info("Prepared npz: %s", npz)

    elif args.cmd == "infer":
        run_inference(
            cfg=cfg,
            model_path=args.model_path,
            npz_path=args.npz,
            metrics_csv=args.metrics_csv,
            code_emb_path=args.code_emb_path,
            out_prefix=args.out_prefix,
            threshold=args.threshold,
        )

    elif args.cmd == "infer-new":
        npz = prepare_new_dataset_from_code(
            cfg,
            out_name=args.out_name,
            granularity=args.granularity,
            include_tests=(args.include_tests.lower() == "true"),
        )
        run_inference(
            cfg=cfg,
            model_path=args.model_path,
            npz_path=npz,
            out_prefix=args.out_name,
            threshold=args.threshold,
        )

if __name__ == "__main__":
    main()