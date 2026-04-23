"""Treina quatro classificadores meta sobre um CSV de medidas de hardness.

Avalia por validação cruzada estratificada de 5 dobras. Uma linha por modelo na saída.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "data" / "datasets"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "meta_results"

META_FILES = {
    ("diabetes", "pyhard"): "diabetes_meta_pyhard_10_20_30pct.csv",
    ("diabetes", "hcat"): "diabetes_meta_hcat_10_20_30pct.csv",
    ("covertype", "pyhard"): "covertype_meta_pyhard_10_20_30pct.csv",
    ("covertype", "hcat"): "covertype_meta_hcat_10_20_30pct.csv",
}

ID_COLS = {"encounter_id", "instance_id"}
TARGET_COL = "problema_qualidade"
N_FOLDS = 5
SEED = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def load_meta(dataset: str, tool: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATASETS_DIR / META_FILES[(dataset, tool)]
    df = pd.read_csv(path)
    y = (df[TARGET_COL] == "sim").astype(int).to_numpy()
    feats = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    X = df[feats].to_numpy(dtype=float)
    return X, y


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    imputer = SimpleImputer(strategy="median")
    models = {
        "Random Forest": Pipeline([
            ("imputer", imputer),
            ("clf", RandomForestClassifier(
                n_estimators=100, random_state=SEED,
                class_weight="balanced", n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", imputer),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, random_state=SEED, class_weight="balanced",
            )),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imputer", imputer),
            ("clf", XGBClassifier(
                n_estimators=100, random_state=SEED,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist", n_jobs=-1,
            )),
        ])
    return models


def run(dataset: str, tool: str, out_path: Path) -> pd.DataFrame:
    X, y = load_meta(dataset, tool)
    n_pos = int(y.sum())
    scale_pos_weight = (len(y) - n_pos) / n_pos if n_pos else 1.0

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for name, pipe in build_models(scale_pos_weight).items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"dataset": dataset, "tool": tool, "model": name, "n_folds": N_FOLDS}
        for metric in SCORING:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"]))
        rows.append(row)

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"{out_path}: {df_out.shape}")
    return df_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["diabetes", "covertype"], required=True)
    ap.add_argument("--tool", choices=["pyhard", "hcat"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or DEFAULT_OUT_DIR / f"meta_cv_{args.dataset}_{args.tool}.csv"
    run(args.dataset, args.tool, out)


if __name__ == "__main__":
    main()
