"""Medidas PyHard sobre o conjunto consolidado do Diabetes, calculadas em lotes.

A matriz de Gower completa (N×N) é inviável para N~1.6e5; este script percorre
o conjunto em blocos de ``BATCH_SIZE`` instâncias e calcula, a cada iteração,
apenas a matriz ``lote × N``. Medidas que exigem a matriz completa (N1 e
Usefulness) não são computadas e aparecem como NaN na saída.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "data" / "datasets"
CONJUNTO_CSV = DATASETS_DIR / "diabetes_conjunto_10_20_30pct.csv"
ORIGINAL_CSV = DATASETS_DIR / "diabetes_original.csv"
OUTPUT_CSV = DATASETS_DIR / "diabetes_meta_pyhard_10_20_30pct.csv"

ID_COL = "encounter_id"
EXCLUDE = {ID_COL, "patient_nbr", "origem", "problema_qualidade"}
BATCH_SIZE = 2000
K_KDN = 10
SEED = 42


def gower_distance(X_query: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    n_feat = X_query.shape[1]
    out = np.zeros((X_query.shape[0], X_ref.shape[0]), dtype=np.float64)
    for j in range(n_feat):
        rng = max(np.ptp(X_ref[:, j]), 1e-8)
        out += pairwise_distances(X_query[:, j : j + 1], X_ref[:, j : j + 1], metric="manhattan") / rng
    return out / n_feat


def batch_measures(dist: np.ndarray, y_full: np.ndarray, y_batch: np.ndarray, k: int):
    n_batch = dist.shape[0]
    idx_sorted = np.argsort(dist, axis=1)
    class_size = {c: np.sum(y_full == c) for c in np.unique(y_full)}

    kdn = np.zeros(n_batch)
    n2 = np.full(n_batch, np.nan)
    lsc = np.full(n_batch, np.nan)
    lsr = np.full(n_batch, np.nan)

    for i in range(n_batch):
        order = idx_sorted[i]
        labels_sorted = y_full[order]
        d_sorted = dist[i, order]

        kdn[i] = np.mean(labels_sorted[1 : k + 1] != y_batch[i])

        same = labels_sorted == y_batch[i]
        same_pos = np.where(same)[0]
        diff_pos = np.where(~same)[0]
        if len(same_pos) < 2 or len(diff_pos) == 0:
            continue
        d_same = d_sorted[same_pos[1]]
        d_enemy = d_sorted[diff_pos[0]]
        n2[i] = d_same / max(d_enemy, 1e-15)
        lsc[i] = 1 - diff_pos[0] / max(class_size.get(y_batch[i], 1), 1)
        last_same = len(labels_sorted) - 1 - np.argmax(same[::-1])
        lsr[i] = 1 - min(1.0, d_enemy / max(d_sorted[last_same], 1e-15))

    n2 = 1 - 1 / (n2 + 1)
    return kdn, n2, lsc, lsr, idx_sorted


def harmfulness(ne_indices: np.ndarray, y_full: np.ndarray) -> np.ndarray:
    n = len(y_full)
    class_count = pd.Series(y_full).value_counts()
    denom = np.array([max(n - class_count.get(y_full[i], 0), 1) for i in range(n)], dtype=float)
    return np.bincount(ne_indices, minlength=n) / denom


def f1_overlap(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, n_feat = X.shape
    classes = np.unique(y)
    pairs = list(combinations(classes, 2))
    score = np.zeros(n)
    for c1, c2 in pairs:
        mask = (y == c1) | (y == c2)
        sub_x, sub_y = X[mask], y[mask]
        for j in range(n_feat):
            f1 = sub_x[sub_y == c1, j]
            f2 = sub_x[sub_y == c2, j]
            if len(f1) == 0 or len(f2) == 0:
                continue
            lo = max(np.min(f1), np.min(f2))
            hi = min(np.max(f1), np.max(f2))
            score += ((X[:, j] >= lo) & (X[:, j] <= hi)).astype(float) / n_feat
    return score / max(len(pairs), 1)


def tree_and_nb_measures(X: np.ndarray, y: np.ndarray):
    dtc = tree.DecisionTreeClassifier(criterion="gini", random_state=SEED).fit(X, y)
    grid = GridSearchCV(
        tree.DecisionTreeClassifier(criterion="gini", random_state=SEED),
        {"ccp_alpha": np.linspace(0.001, 0.1, num=20)},
        n_jobs=-1,
    ).fit(X, y)
    dtc_pruned = tree.DecisionTreeClassifier(
        criterion="gini", ccp_alpha=grid.best_params_["ccp_alpha"], random_state=SEED
    ).fit(X, y)

    leaves = dtc_pruned.apply(X)
    leaf_sizes = pd.Series(leaves).value_counts()
    same_class = np.array([np.sum((leaves == leaves[i]) & (y == y[i])) for i in range(len(X))])
    dcp = 1 - same_class / np.array([leaf_sizes[l] for l in leaves])

    td_u = (np.asarray(dtc.decision_path(X).sum(axis=1)).ravel() - 1) / max(dtc.get_depth(), 1)
    td_p = (np.asarray(dtc_pruned.decision_path(X).sum(axis=1)).ravel() - 1) / max(
        dtc_pruned.get_depth(), 1
    )

    n_c = len(np.unique(y))
    nb = CalibratedClassifierCV(
        GaussianNB(priors=np.ones(n_c) / n_c),
        method="sigmoid",
        cv=3,
        ensemble=False,
        n_jobs=-1,
    ).fit(X, y)
    proba = nb.predict_proba(X)
    cl_idx = np.argmax(y.reshape(-1, 1) == nb.classes_, axis=1)
    cl = 1 - proba[np.arange(len(X)), cl_idx]
    p_true = proba[np.arange(len(X)), cl_idx]
    p_other = np.max(
        np.where(np.arange(proba.shape[1])[None, :] != cl_idx[:, None], proba, -np.inf), axis=1
    )
    cld = (1 - (p_true - p_other)) / 2
    return dcp, td_u, td_p, cl, cld


def load_features(conjunto_csv: Path, original_csv: Path):
    conjunto = pd.read_csv(conjunto_csv)
    original = pd.read_csv(original_csv).set_index(ID_COL)["y"]
    conjunto["y"] = conjunto[ID_COL].map(original)
    conjunto = conjunto.dropna(subset=["y"]).reset_index(drop=True)
    conjunto["y"] = conjunto["y"].astype(int)

    feature_cols = [c for c in conjunto.columns if c not in EXCLUDE and c != "y"]
    X_df = conjunto[feature_cols].copy()
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.Categorical(X_df[col].astype(str)).codes

    valid = X_df.notna().all(axis=1)
    X = X_df.loc[valid].to_numpy(dtype=np.float64)
    y = conjunto.loc[valid, "y"].to_numpy(dtype=int)
    ids = conjunto.loc[valid, ID_COL].to_numpy()
    flag = conjunto.loc[valid, "problema_qualidade"].to_numpy()
    return X, y, ids, flag


def main():
    X, y, ids, flag = load_features(CONJUNTO_CSV, ORIGINAL_CSV)
    n = X.shape[0]

    dcp, td_u, td_p, cl, cld = tree_and_nb_measures(X, y)
    f1 = f1_overlap(X, y)

    kdn_all = np.empty(n)
    n2_all = np.empty(n)
    lsc_all = np.empty(n)
    lsr_all = np.empty(n)
    ne_idx = np.zeros(n, dtype=int)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        dist = gower_distance(X[start:end], X)
        for i in range(end - start):
            dist[i, start + i] = np.inf
        kdn, n2, lsc, lsr, order = batch_measures(dist, y, y[start:end], k=K_KDN)
        kdn_all[start:end] = kdn
        n2_all[start:end] = n2
        lsc_all[start:end] = lsc
        lsr_all[start:end] = lsr
        for i in range(end - start):
            labels_sorted = y[order[i]]
            diff = np.where(labels_sorted != y[start + i])[0]
            if len(diff):
                ne_idx[start + i] = order[i, diff[0]]

    out = pd.DataFrame({
        ID_COL: ids,
        "problema_qualidade": flag,
        "kDN": kdn_all,
        "DCP": dcp,
        "TD_P": td_p,
        "TD_U": td_u,
        "CL": cl,
        "CLD": cld,
        "N1": np.nan,
        "N2": n2_all,
        "LSC": lsc_all,
        "LSR": lsr_all,
        "Harmfulness": harmfulness(ne_idx, y),
        "Usefulness": np.nan,
        "F1": f1,
    })
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"{OUTPUT_CSV}: {out.shape}")


if __name__ == "__main__":
    main()
