"""Extrai medidas H-CAT por instância sobre o conjunto consolidado do Diabetes.

Treina o MLP tabular de referência (sem perturbação adicional; o ruído já está
no conjunto) e coleta os escores de cada método de caracterização. Requer o
repositório H-CAT clonado em ``TCC/H-CAT``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
HCAT_DIR = REPO_ROOT / "H-CAT"
if not HCAT_DIR.is_dir():
    raise FileNotFoundError(f"H-CAT não encontrado em {HCAT_DIR}")
sys.path.insert(0, str(HCAT_DIR))

from src.dataloader import MultiFormatDataLoader  # noqa: E402
from src.models import MLP  # noqa: E402
from src.trainer import PyTorchTrainer  # noqa: E402
from src.utils import seed_everything  # noqa: E402

DATASETS_DIR = REPO_ROOT / "data" / "datasets"
CONJUNTO_CSV = DATASETS_DIR / "diabetes_conjunto_10_20_30pct.csv"
ORIGINAL_CSV = DATASETS_DIR / "diabetes_original.csv"
OUTPUT_CSV = DATASETS_DIR / "diabetes_meta_hcat_10_20_30pct.csv"

ID_COL = "encounter_id"
EXCLUDE = {ID_COL, "patient_nbr", "origem", "problema_qualidade"}
LABEL_MAP = {"NO": 0, ">30": 1, "<30": 2}
METHODS = [
    "aum", "data_uncert", "el2n", "grand", "cleanlab",
    "forgetting", "vog", "prototypicality", "loss", "conf_agree",
]

SEED = 42
BATCH_SIZE = 256
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features():
    conjunto = pd.read_csv(CONJUNTO_CSV)
    original = pd.read_csv(ORIGINAL_CSV)[[ID_COL, "readmitted"]].drop_duplicates()
    if not pd.api.types.is_numeric_dtype(original["readmitted"]):
        original["readmitted"] = original["readmitted"].map(LABEL_MAP)
    original = original.rename(columns={"readmitted": "y"})

    conjunto = conjunto.merge(original, on=ID_COL, how="left").dropna(subset=["y"])
    conjunto["y"] = conjunto["y"].astype(int)

    feature_cols = [c for c in conjunto.columns if c not in EXCLUDE and c != "y"]
    X_df = conjunto[feature_cols].copy()
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.Categorical(X_df[col].astype(str)).codes

    valid = X_df.notna().all(axis=1)
    X_df = X_df.loc[valid].reset_index(drop=True)
    conjunto = conjunto.loc[valid].reset_index(drop=True)
    X = X_df.to_numpy(dtype=np.float32)
    y = conjunto["y"].to_numpy(dtype=int)
    return X, y, conjunto


def to_flat(scores) -> np.ndarray:
    if isinstance(scores, (list, tuple)):
        if not scores:
            return np.array([])
        if hasattr(scores[0], "shape"):
            return np.concatenate([np.asarray(x).ravel() for x in scores])
    return np.asarray(scores).ravel()


def extract_scores(hardness: dict) -> dict[str, np.ndarray]:
    out = {}
    for method, obj in hardness.items():
        if method in ("dataiq", "datamaps"):
            _, scores = obj.compute_scores(datamaps=(method == "datamaps"))
        elif method == "cleanlab":
            scores, _ = obj.compute_scores()
        else:
            if getattr(obj, "_scores", None) is None:
                obj.compute_scores()
            scores = obj.scores
        arr = to_flat(scores)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=np.nanmean(arr))
        out[method] = arr
    return out


def main():
    seed_everything(SEED)
    X, y, conjunto = load_features()
    n_classes = len(np.unique(y))

    loader_factory = MultiFormatDataLoader(
        data=(X, y),
        target_column=None,
        data_type="numpy",
        data_modality="tabular",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        transform=None,
        image_transform=None,
        perturbation_method="uniform",
        p=0.0,
        rule_matrix=None,
        atypical_marginal=[],
    )
    loader, loader_unshuffled = loader_factory.get_dataloader()

    model = MLP(input_size=X.shape[1], nlabels=n_classes).to(DEVICE)
    trainer = PyTorchTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        lr=1e-3,
        epochs=EPOCHS,
        total_samples=X.shape[0],
        num_classes=n_classes,
        device=DEVICE,
        characterization_methods=METHODS,
    )
    trainer.fit(loader, loader_unshuffled)

    scores = extract_scores(trainer.get_hardness_methods())
    meta = pd.concat(
        [conjunto[[ID_COL, "problema_qualidade"]].reset_index(drop=True), pd.DataFrame(scores)],
        axis=1,
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUTPUT_CSV, index=False)
    print(f"{OUTPUT_CSV}: {meta.shape}")


if __name__ == "__main__":
    main()
