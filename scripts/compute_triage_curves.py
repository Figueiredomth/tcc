"""Gera as curvas de triagem (Precision@k, Recall@k, Lift@k) dos meta-classificadores.

Para cada trilha (PyHard, H-CAT) e cada modelo (RF, GB, LR, XGB), produz scores
in-domain (out-of-fold) nos dois conjuntos e scores de transferência
Diabetes → Covertype, e calcula as métricas em k ∈ {5%, 10%, 20%, 30%}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "data" / "datasets"
DEFAULT_OUT = REPO_ROOT / "data" / "meta_results" / "triage_curves.csv"

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
K_FRACTIONS = [0.05, 0.10, 0.20, 0.30]


def load_Xy(path: Path):
    df = pd.read_csv(path)
    y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "sim").astype(int).to_numpy()
    feats = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return X, y, feats


def build_pipeline(name: str, scale_pos_weight: float) -> Pipeline:
    imputer = SimpleImputer(strategy="median")
    if name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=100, random_state=SEED,
            class_weight="balanced", n_jobs=-1,
        )
        return Pipeline([("imputer", imputer), ("clf", clf)])
    if name == "Gradient Boosting":
        return Pipeline([
            ("imputer", imputer),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
        ])
    if name == "Logistic Regression":
        return Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, random_state=SEED, class_weight="balanced",
            )),
        ])
    if name == "XGBoost":
        return Pipeline([
            ("imputer", imputer),
            ("clf", XGBClassifier(
                n_estimators=100, random_state=SEED,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist", n_jobs=-1,
            )),
        ])
    raise ValueError(f"Modelo desconhecido: {name}")


def triage_metrics(scores: np.ndarray, y_true: np.ndarray, k_frac: float) -> dict:
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    top_idx = np.argsort(-scores)[:k]
    tp = int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())
    recall_k = tp / total_pos if total_pos else float("nan")
    return {
        "k_frac": k_frac,
        "k": k,
        "n": n,
        "tp_at_k": tp,
        "precision_at_k": tp / k,
        "recall_at_k": recall_k,
        "lift_at_k": recall_k / k_frac if k_frac else float("nan"),
        "prevalence": float(y_true.mean()),
    }


def run_indomain(dataset: str, tool: str, models: list[str]) -> list[dict]:
    X, y, _ = load_Xy(DATASETS_DIR / META_FILES[(dataset, tool)])
    n_pos = int(y.sum())
    spw = (len(y) - n_pos) / n_pos if n_pos else 1.0
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows = []
    for name in models:
        proba = cross_val_predict(
            build_pipeline(name, spw), X, y, cv=cv, method="predict_proba", n_jobs=1,
        )[:, 1]
        for k_frac in K_FRACTIONS:
            rows.append({
                **triage_metrics(proba, y, k_frac),
                "setting": "in_domain",
                "train_dataset": dataset,
                "test_dataset": dataset,
                "tool": tool,
                "model": name,
            })
    return rows


def run_transfer(tool: str, models: list[str]) -> list[dict]:
    X_tr_df = pd.read_csv(DATASETS_DIR / META_FILES[("diabetes", tool)])
    X_te_df = pd.read_csv(DATASETS_DIR / META_FILES[("covertype", tool)])
    feats_tr = [c for c in X_tr_df.columns if c not in ID_COLS and c != TARGET_COL]
    feats_te = [c for c in X_te_df.columns if c not in ID_COLS and c != TARGET_COL]
    common = [c for c in feats_tr if c in feats_te]

    y_tr = (X_tr_df[TARGET_COL].astype(str).str.strip().str.lower() == "sim").astype(int).to_numpy()
    y_te = (X_te_df[TARGET_COL].astype(str).str.strip().str.lower() == "sim").astype(int).to_numpy()
    X_tr = X_tr_df[common].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X_te = X_te_df[common].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    n_pos = int(y_tr.sum())
    spw = (len(y_tr) - n_pos) / n_pos if n_pos else 1.0

    rows = []
    for name in models:
        pipe = build_pipeline(name, spw)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        for k_frac in K_FRACTIONS:
            rows.append({
                **triage_metrics(proba, y_te, k_frac),
                "setting": "transfer",
                "train_dataset": "diabetes",
                "test_dataset": "covertype",
                "tool": tool,
                "model": name,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    models = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    if HAS_XGB:
        models.append("XGBoost")

    rows = []
    for tool in ("pyhard", "hcat"):
        rows += run_indomain("diabetes", tool, models)
        rows += run_indomain("covertype", tool, models)
        rows += run_transfer(tool, models)

    cols = [
        "setting", "train_dataset", "test_dataset", "tool", "model",
        "k_frac", "k", "n", "tp_at_k",
        "precision_at_k", "recall_at_k", "lift_at_k", "prevalence",
    ]
    df = pd.DataFrame(rows)[cols].round({"precision_at_k": 4, "recall_at_k": 4, "lift_at_k": 4})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{args.out}: {df.shape}")


if __name__ == "__main__":
    main()
