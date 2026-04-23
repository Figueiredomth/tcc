"""
Baixa o dataset Diabetes 130-US Hospitals do OpenML e grava em CSV.

Saída: data/diabetes_full.csv
"""

from pathlib import Path

import pandas as pd
import openml

DATASET_ID = 4541
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "diabetes_full.csv"


def main():
    dataset = openml.datasets.get_dataset(DATASET_ID)
    X, y, _, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )

    df = pd.DataFrame(X, columns=attribute_names)
    df["readmitted"] = y

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Linhas: {len(df):,}, colunas: {len(df.columns)} (inclui readmitted)")
    print(f"Classes (readmitted): {sorted(df['readmitted'].astype(str).unique().tolist())}")
    print(f"Salvo: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
