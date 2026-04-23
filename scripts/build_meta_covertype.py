"""Gera os blocos A (original) e B (perturbado) para o Forest Covertype, para 10%, 20% e 30% de ruído.

Por padrão usa o Covertype completo. ``--max-rows N`` aplica amostra estratificada
para caber em memória ao calcular PyHard em lotes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "data" / "covertype_full.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "datasets"
ID_COL = "instance_id"
TARGET_COL = "y"
PROPORTIONS = [0.1, 0.2, 0.3]
SEED = 42


def uniform_mislabeling(labels: np.ndarray, p: float, rng: np.random.Generator):
    n = len(labels)
    flip_idx = rng.choice(n, size=int(round(p * n)), replace=False)
    classes = np.unique(labels)
    flipped = labels.copy()
    for i in flip_idx:
        others = classes[classes != labels[i]]
        flipped[i] = rng.choice(others)
    return flip_idx, flipped


def main(max_rows: int | None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV).dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    if max_rows is not None and len(df) > max_rows:
        idx, _ = train_test_split(
            df.index, train_size=max_rows, stratify=df[TARGET_COL], random_state=SEED
        )
        df = df.loc[idx].reset_index(drop=True)

    df.insert(0, ID_COL, np.arange(len(df), dtype=np.int64))
    labels = df[TARGET_COL].to_numpy()
    df.to_csv(OUTPUT_DIR / "covertype_original.csv", index=False)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    block_a = df[feature_cols].assign(problema_qualidade="nao", origem="A")
    block_a.to_csv(OUTPUT_DIR / "covertype_block_a.csv", index=False)

    rng = np.random.default_rng(SEED)
    blocks_b = []
    for p in PROPORTIONS:
        flip_idx, flipped = uniform_mislabeling(labels, p, rng)
        block_b = df.loc[flip_idx, feature_cols].copy()
        block_b[TARGET_COL] = flipped[flip_idx]
        block_b["problema_qualidade"] = "sim"
        block_b["origem"] = "B"
        suffix = f"{int(p * 100)}pct"
        block_b.to_csv(OUTPUT_DIR / f"covertype_block_b_{suffix}.csv", index=False)
        blocks_b.append(block_b.drop(columns=[TARGET_COL]))

        conjunto = pd.concat([block_a, block_b.drop(columns=[TARGET_COL])], ignore_index=True)
        conjunto.to_csv(OUTPUT_DIR / f"covertype_conjunto_{suffix}.csv", index=False)

    conjunto_completo = pd.concat([block_a, *blocks_b], ignore_index=True)
    conjunto_completo.to_csv(OUTPUT_DIR / "covertype_conjunto_10_20_30pct.csv", index=False)
    print(f"{OUTPUT_DIR}: blocos A e B ({PROPORTIONS}) e conjunto consolidado (N={len(df)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    main(args.max_rows)
