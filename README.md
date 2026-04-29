# FOMC tone and semantic shifts

Projet NLP sur deux types de documents du FOMC :

- `Statements` : communiqués courts publiés au moment de la décision ;
- `Minutes` : comptes rendus plus longs publiés après la réunion.

Le dépôt garde les données collectées aussi simples que possible. Les nettoyages, labels et variables exploratoires sont construits dans le notebook pour que les étapes soient visibles.

## Données

Sources :

- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

Période par défaut : 2000-2024.

### CSV versionnés

`data/raw/fomc_meeting_links.csv`

- une ligne par réunion trouvée sur le site de la Fed ;
- colonnes : `date`, `year`, `meeting_id`, `statement_url`, `minutes_url` ;
- pas de texte ;
- pas de nettoyage ;
- pas de label.

`data/raw/fomc_documents_raw.csv`

- une ligne par réunion ;
- colonnes principales : `date`, `year`, `meeting_id`, `statement_url`, `minutes_url`, `statement_text`, `minutes_text` ;
- les textes sont les textes extraits des pages HTML de la Fed ;
- pas de `rate_decision` ;
- pas de `clean_text` ;
- pas de score hawkish/dovish ;
- pas de TF-IDF ;
- pas de variables pré-calculées.

C'est le fichier à utiliser pour repartir sur une autre approche NLP.

### Ce qui n'est plus versionné

Les anciens CSV de `data/processed/` ont été retirés. Ils mélangeaient données et premiers calculs, ce qui rendait moins clair ce qui venait de la collecte et ce qui venait de l'analyse.

Si on veut des longueurs, des textes nettoyés, des labels `hike/cut/hold`, des scores lexicaux ou des similarités TF-IDF, on les calcule dans le notebook.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ml_for_nlp --display-name "Python (ml_for_nlp)"
```

## Collecte

```bash
python scripts/run_pipeline.py --start-year 2000 --end-year 2024
```

La collecte écrit seulement :

```text
data/raw/fomc_meeting_links.csv
data/raw/fomc_documents_raw.csv
```

Sur la période 2000-2024, on obtient 199 réunions avec Statement et Minutes.

## Notebook

```text
notebooks/fomc_document_analysis.ipynb
```

Kernel à sélectionner : `Python (ml_for_nlp)`.

Le notebook repart de `data/raw/fomc_documents_raw.csv`. Les étapes y sont écrites directement :

- nettoyage minimal du texte ;
- comptage des mots ;
- première règle simple pour `rate_decision` ;
- dictionnaires hawkish/dovish et incertitude ;
- TF-IDF pour les ruptures entre documents successifs ;
- TF-IDF pour comparer Statement et Minutes d'une même réunion.

## Organisation

```text
src/fomc_nlp/
  data_collection.py   # scraping Fed et extraction texte
  pipeline.py          # CLI de collecte

scripts/
  run_pipeline.py

notebooks/
  fomc_document_analysis.ipynb
```
