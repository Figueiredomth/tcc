"""
Baixa o dataset Covertype **completo** do OpenML (ID 1596) e grava em CSV.

Saída: data/covertype_full.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import openml

DATASET_ID = 1596
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "covertype_full.csv"


def main():
    dataset = openml.datasets.get_dataset(DATASET_ID)
    X, y, _, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )

    df = pd.DataFrame(X, columns=attribute_names)
    df["y"] = y
    df = df.dropna()

    le = LabelEncoder()
    df["y"] = le.fit_transform(df["y"].astype(str))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Linhas: {len(df):,}, colunas: {len(df.columns)} (inclui y)")
    print(f"Classes (y): {sorted(df['y'].unique().tolist())}")
    print(f"Salvo: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
