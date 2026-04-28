# FOMC tone and semantic shifts

Projet NLP sur la communication du FOMC. On compare deux corpus publics de la Federal Reserve :

- les **Statements**, courts et centrés sur la décision ;
- les **Minutes**, plus longues, plus détaillées, publiées après la réunion.

L'idée est de transformer ces textes en indicateurs simples : ton hawkish/dovish, incertitude, ruptures lexicales dans le temps, proximité entre Statement et Minutes, puis prédiction de la décision de taux (`hike`, `cut`, `hold`).

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
- `data/processed/model_results.csv`
- `data/processed/top_semantic_shifts.csv`
- `data/processed/misclassified_examples.csv`

Les labels `rate_decision` sont inférés à partir des Statements. Les corrections manuelles peuvent être mises dans `data/raw/rate_decisions_manual.csv`.

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
python scripts/run_pipeline.py --steps models
```

Vérifier rapidement les CSV produits :

```bash
python scripts/check_data_sanity.py
```

## Organisation

```text
src/fomc_nlp/
  data_collection.py   # URLs Fed, textes, fusion, labels
  preprocessing.py     # nettoyage et masquage des phrases de décision
  features.py          # features, à developer, compléter ...
  visualization.py     # graphiques
  modeling.py          # baseline, logreg, SVM
  evaluation.py        # métriques et erreurs
  pipeline.py          # CLI
```

## Méthodes incluses

À compléter

## Points à discuter dans le rapport

- Différence de typologie de document : pas le même format, role institutionel 
- Analyse (à compléter)
