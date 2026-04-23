# TCC — Avaliação e combinação de medidas de *instance hardness* para detecção de ruído em rótulos

Comparação empírica das ferramentas **H-CAT** (medidas dinâmicas, extraídas da dinâmica de treino de uma MLP) e **PyHard** (medidas estáticas de complexidade da instância) em dois datasets tabulares, e uso das medidas como atributos de um classificador meta que prediz se uma amostra está perturbada.

## Datasets

- **Diabetes 130-US Hospitals** — OpenML ID `4541` (~101k instâncias, 47 features, 3 classes)
- **Forest Covertype** — OpenML ID `1596` (~581k instâncias, 54 features, 7 classes)

Ambos são baixados automaticamente via `scripts/fetch_*_full_openml.py`.

## Estrutura do repositório

```
projeto/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── datasets/                     # Meta-datasets (saída do pipeline)
│   │   ├── *_meta_pyhard_10_20_30pct.csv
│   │   └── *_meta_hcat_10_20_30pct.csv
│   ├── meta_results/                 # CVs e tabelas dos classificadores meta
│   └── comparison_results/           # Tabelas e figuras da comparação H-CAT × PyHard
│
├── scripts/                          # Pipeline em Python puro (ver seção abaixo)
├── notebooks/                        # Replicações e análises (ver notebooks/README.md)
├── docs/                             # Monografia e documentação auxiliar
│
├── H-CAT/                            # Submódulo: implementação de referência do H-CAT
└── pyhard/                           # Submódulo: implementação de referência do PyHard
```

Os submódulos `H-CAT/` e `pyhard/` precisam estar presentes antes de rodar o pipeline. Se o repositório foi clonado sem eles, obtenha-os em:

- H-CAT: <https://github.com/seedatnabeel/H-CAT>
- PyHard: <https://gitlab.com/ita-ml/pyhard>

## Instalação

```bash
pip install -r requirements.txt
```

`torch` precisa corresponder à sua GPU/CUDA; consulte <https://pytorch.org> caso a instalação padrão não detecte GPU.

## Pipeline de reprodução

A ordem abaixo regenera todos os artefatos usados na monografia. Cada etapa pode ser pulada se o artefato de saída já existe.

```bash
# 1. Baixar os datasets brutos
python scripts/fetch_diabetes_full_openml.py
python scripts/fetch_covertype_full_openml.py

# 2. Fase 0 — gerar blocos A (original) e B (perturbado) com 10/20/30% de ruído
python scripts/build_diabetes_meta_datasets.py
python scripts/build_covertype_meta_datasets.py --max-rows 20000

# 3. Calcular medidas de instance hardness (principal gargalo computacional)
python scripts/compute_pyhard_measures_batched.py
python scripts/compute_pyhard_measures_batched_covertype.py --batch-size 2000
python scripts/compute_hcat_measures_meta.py
python scripts/compute_hcat_measures_meta_covertype.py

# 4. Treinar classificadores meta (in-domain) e avaliar transferência
python scripts/train_meta_classifiers.py
python scripts/train_meta_transfer.py

# 5. Curvas de triagem (Lift@k, Precision@k, Recall@k)
python scripts/compute_triage_curves.py
```

Os notebooks em `notebooks/` consumem os artefatos gerados e produzem as figuras e tabelas do TCC. A ordem de execução sugerida está em `notebooks/README.md`.

### Execução sem ambiente local

`notebooks/colab/Covertype_Colab_standalone.ipynb` reproduz o pipeline completo do Covertype em um único notebook para o Google Colab (clona os submódulos, baixa o dataset, roda as etapas 1-3).

## Configuração do experimento

- **Perturbações** (cinco tipos do paper H-CAT): `uniform`, `asymmetric`, `adjacent`, `instance`, `atypical`.
- **Proporções de ruído**: 10%, 20%, 30%.
- **Runs por configuração**: 3 (seeds 0, 1, 2).
- **Seed principal**: `SEED = 42` (build/treino), `SEED = 0` (replicações H-CAT/PyHard, alinhado ao paper).

## Dados versionados e dados reproduzíveis

Por tamanho, nem todos os CSVs vão no git. A regra está em `.gitignore`:

| Artefato | Em `.gitignore`? | Como obter |
|---|---|---|
| `data/*_full.csv` (raw) | Sim | `scripts/fetch_*_full_openml.py` |
| `data/datasets/*_block_*.csv`, `*_original.csv`, `*_conjunto_{10,20,30}pct.csv` | Sim | `scripts/build_*_meta_datasets.py` |
| `data/datasets/*_conjunto_10_20_30pct.csv` | Não | versionado (é entrada de `compute_*`) |
| `data/datasets/*_meta_*_10_20_30pct.csv` | Não | versionado (é a saída cara do pipeline) |
| `data/meta_results/*.csv` | Não | versionado (saída de `train_meta_*`) |
| `data/comparison_results/*.csv`, `figures/*.png` | Não | versionado (saída do notebook 04) |


## Referências

- Seedat, N.; Imrie, F.; van der Schaar, M. **Dissecting Sample Hardness: A Fine-Grained Analysis of Hardness Characterization Methods for Data-Centric AI** (ICLR 2024).
- Paiva, P. Y. A. et al. **PyHard: A novel tool for generating hardness embeddings to support data-centric analysis** (2023).
