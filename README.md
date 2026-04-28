# FOMC tone and semantic shifts

Projet NLP sur la communication du FOMC. On compare deux corpus publics de la Federal Reserve :

- les **Statements**, courts et centrés sur la décision ;
- les **Minutes**, plus longues, plus détaillées, publiées après la réunion.

L'idée est de transformer ces textes en indicateurs simples : ton hawkish/dovish, incertitude, ruptures lexicales dans le temps, et proximité entre Statement et Minutes.

## Données

Sources :

- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

Période par défaut : 2000-2024. Après collecte, on obtient 199 réunions appariées Statement-Minutes.

### Ce qu'il y a dans les CSV

`data/raw/fomc_meeting_links.csv`

- Sortie de collecte des liens.
- Une ligne par réunion détectée sur le site de la Fed.
- Contient seulement `date`, `year`, `meeting_id`, `statement_url`, `minutes_url`.
- Sert surtout à vérifier que le scraping a trouvé les bons documents.
- Ne contient aucun texte.

`data/raw/fomc_documents_raw.csv`

- Fichier à utiliser si on veut repartir sur une autre approche NLP.
- Une ligne par réunion.
- Contient les URLs et les textes extraits des pages HTML officielles : `statement_text` et `minutes_text`.
- Ne contient pas de texte nettoyé, pas de score lexical, pas de TF-IDF, pas de label de décision.
- Ce n'est pas le HTML brut complet : c'est le texte extrait de la page Fed.

`data/processed/fomc_statements.csv`

- Une ligne par Statement.
- Contient `statement_text`, `statement_clean_text`, `statement_n_words`.
- Ajoute `rate_decision`, inféré automatiquement à partir du Statement (`hike`, `cut`, `hold`).
- Utile si on veut travailler seulement sur les Statements avec un nettoyage simple déjà fait.

`data/processed/fomc_minutes.csv`

- Une ligne par compte rendu de Minutes.
- Contient `minutes_text`, `minutes_clean_text`, `minutes_n_words`.
- Ne contient pas de `rate_decision`, car la décision est portée par le Statement dans ce pipeline.
- Utile si on veut travailler seulement sur les Minutes.

`data/processed/fomc_merged.csv`

- Une ligne par réunion, avec Statement et Minutes sur la même ligne.
- Contient les textes originaux extraits, les textes nettoyés, les nombres de mots et `rate_decision`.
- C'est le fichier pratique pour faire une analyse comparative simple.
- Il ne contient pas encore les scores d'analyse.

`data/processed/fomc_features.csv`

- Même unité que `fomc_merged.csv` : une ligne par réunion.
- Contient déjà les variables d'analyse du projet :
  - scores hawkish/dovish ;
  - scores d'incertitude ;
  - textes masqués pour les phrases de décision ;
  - similarité TF-IDF entre documents successifs ;
  - distance TF-IDF entre Statement et Minutes.
- Ce fichier n'est pas un point de départ brut. Il sert à faire les graphiques et les premières interprétations.

`data/processed/keyword_frequencies.csv`

- Format long : une ligne par date, type de document et mot-clé.
- Colonnes principales : `date`, `document_type`, `keyword`, `count`, `frequency`.
- Sert aux graphiques de fréquence de mots-clés dans le temps.

`data/processed/top_semantic_shifts.csv`

- Petit fichier de lecture.
- Contient les plus grands pics de rupture TF-IDF, séparément pour Statements et Minutes.
- Inclut la date, le type de document, la décision de taux, les scores associés et un extrait court.

En résumé :

- pour refaire une autre analyse NLP : partir de `data/raw/fomc_documents_raw.csv` ;
- pour une analyse simple avec textes nettoyés et labels : partir de `data/processed/fomc_merged.csv` ;
- pour reproduire les graphiques déjà prévus : utiliser `data/processed/fomc_features.csv` et les deux fichiers dérivés.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ml_for_nlp --display-name "Python (ml_for_nlp)"
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

`collect` produit les fichiers `data/raw/` et les CSV de base dans `data/processed/`. `features` ajoute les scores et similarités. `figures` ne modifie pas les données.

Vérifier rapidement les CSV produits :

```bash
python scripts/check_data_sanity.py
```

Notebook de première lecture :

```text
notebooks/fomc_document_analysis.ipynb
```

Kernel à sélectionner : `Python (ml_for_nlp)`.

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

Le pipeline ne fait pas de modèle supervisé dans cette version. Il s'arrête à l'analyse descriptive des documents :

- longueur des Statements et des Minutes ;
- dictionnaires simples hawkish/dovish et incertitude ;
- TF-IDF pour mesurer les changements d'un document au suivant ;
- TF-IDF pour comparer Statement et Minutes d'une même réunion.

## Points à discuter dans le rapport

- Les Minutes sont beaucoup plus longues que les Statements.
- Les deux textes n'ont pas le même rôle institutionnel.
- Les scores lexicaux dépendent du dictionnaire retenu.
- TF-IDF mesure d'abord des changements de vocabulaire.
