"""Transferência entre domínios: treina o meta-classificador no Diabetes e avalia no Covertype.

Utiliza apenas as colunas de medidas presentes em ambos os conjuntos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
SEED = 42


def load_Xy(path: Path):
    df = pd.read_csv(path)
    y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "sim").astype(int).to_numpy()
    feats = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    return df, y, feats


def build_classifier(name: str, scale_pos_weight: float):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=SEED,
            class_weight="balanced", n_jobs=-1,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000, random_state=SEED, class_weight="balanced",
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=100, random_state=SEED,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", tree_method="hist", n_jobs=-1,
        )
    raise ValueError(f"Modelo desconhecido: {name}")


def fit_and_score(name: str, X_tr, y_tr, X_te, y_te, scale_pos_weight: float):
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    if name == "Logistic Regression":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    clf = build_classifier(name, scale_pos_weight)

    if name == "Gradient Boosting":
        # sklearn não expõe class_weight no GB; emulamos com sample_weight "balanced".
        counts = np.bincount(y_tr, minlength=2)
        weights = len(y_tr) / (len(counts) * np.maximum(counts, 1))
        clf.fit(X_tr, y_tr, sample_weight=weights[y_tr])
    else:
        clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "average_precision": float(average_precision_score(y_te, proba)),
    }


def run(tool: str, out_path: Path) -> pd.DataFrame:
    df_tr, y_tr, feats_tr = load_Xy(DATASETS_DIR / META_FILES[("diabetes", tool)])
    df_te, y_te, feats_te = load_Xy(DATASETS_DIR / META_FILES[("covertype", tool)])
    common = [c for c in feats_tr if c in feats_te]
    X_tr = df_tr[common].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X_te = df_te[common].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    n_pos = int(y_tr.sum())
    scale_pos_weight = (len(y_tr) - n_pos) / n_pos if n_pos else 1.0

    model_names = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    if HAS_XGB:
        model_names.append("XGBoost")

    rows = [
        {"tool": tool, "n_features": len(common),
         **fit_and_score(name, X_tr, y_tr, X_te, y_te, scale_pos_weight)}
        for name in model_names
    ]
    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"{out_path}: {df_out.shape}")
    return df_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", choices=["pyhard", "hcat"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or DEFAULT_OUT_DIR / f"meta_transfer_{args.tool}.csv"
    run(args.tool, out)


if __name__ == "__main__":
    main()
