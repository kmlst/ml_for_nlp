# FOMC tone and semantic shifts

Projet NLP sur deux types de documents du FOMC :

- `Statements` : communiqués courts publiés au moment de la décision ;
- `Minutes` : comptes rendus plus longs publiés après la réunion.

Sources :

- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

Période : 2000-2024.

### CSV versionnés

`data/raw/fomc_meeting_links.csv`

- une ligne par réunion trouvée sur le site de la Fed ;
- colonnes : `date`, `year`, `meeting_id`, `statement_url`, `minutes_url` ;
`data/raw/fomc_documents_raw.csv`

- une ligne par réunion ;
- colonnes principales : `date`, `year`, `meeting_id`, `statement_url`, `minutes_url`, `statement_text`, `minutes_text` ;
- les textes sont les textes extraits des pages HTML de la Fed ;

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

écrit les fichiers :

```text
data/raw/fomc_meeting_links.csv
data/raw/fomc_documents_raw.csv
```

Sur la période 2000-2024, on obtient 199 réunions avec Statement et Minutes.

## Notebook

```text
notebooks/fomc_document_analysis.ipynb
```

Le notebook repart de `data/raw/fomc_documents_raw.csv`. 


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
