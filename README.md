# FOMC tone and semantic shifts

Projet NLP sur la communication du FOMC. On compare deux corpus publics de la Federal Reserve :

- les **Statements**, courts et centrés sur la décision ;
- les **Minutes**, plus longues, plus détaillées, publiées après la réunion.

L'idée est de transformer ces textes en indicateurs simples : ton hawkish/dovish, incertitude, ruptures lexicales dans le temps, et proximité entre Statement et Minutes.

## Données

Sources :

- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

Période par défaut : 2000-2024.

Fichiers produits :

- `data/processed/fomc_statements.csv`
- `data/processed/fomc_minutes.csv`
- `data/processed/fomc_merged.csv`
- `data/processed/fomc_features.csv`
- `data/processed/top_semantic_shifts.csv`

Les labels `rate_decision` sont inférés automatiquement à partir des Statements.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

Tout lancer :

```bash
python scripts/run_pipeline.py --start-year 2000 --end-year 2024 --steps all
```

Lancer étape par étape :

```bash
python scripts/run_pipeline.py --steps collect
python scripts/run_pipeline.py --steps features
python scripts/run_pipeline.py --steps figures
```

Vérifier rapidement les CSV produits :

```bash
python scripts/check_data_sanity.py
```

Notebook de première lecture :

```text
notebooks/fomc_document_analysis.ipynb
```

## Organisation

```text
src/fomc_nlp/
  data_collection.py   # URLs Fed, textes, fusion, labels
  preprocessing.py     # nettoyage et masquage des phrases de décision
  features.py          # scores lexicaux, TF-IDF, similarités
  visualization.py     # graphiques
  pipeline.py          # CLI
```

## Méthodes incluses

- statistiques de longueur par corpus ;
- scores hawkish/dovish normalisés par nombre de mots ;
- score d'incertitude et de risque ;
- rupture TF-IDF entre deux documents successifs ;
- distance TF-IDF entre le Statement et les Minutes d'une même réunion.

## Points à discuter dans le rapport

- Les Minutes sont beaucoup plus longues que les Statements.
- Les deux textes n'ont pas le même rôle institutionnel.
- Les scores lexicaux dépendent du dictionnaire retenu.
- TF-IDF mesure d'abord des changements de vocabulaire.
