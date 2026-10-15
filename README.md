# Projet Data Mining — Émissions de CO₂

Analyse des émissions mondiales de CO₂ (données Our World in Data) et application web Flask de prédiction par pays jusqu'en 2040.

## Contenu

- **Notebooks** : nettoyage, visualisation, corrélations, clustering des pays et modèle multivarié.
- **Application web** (`app/`) : choix d'un pays et d'une année, prédiction du CO₂ par habitant avec deux modèles (régression polynomiale de degré 2 et Random Forest), trois scénarios (Continuité, Transition énergétique, Forte croissance) et comparaison de plusieurs pays.

## Structure du dépôt

```
projet-data-mining/
├── app/
│   ├── app.py              # application Flask (routes /, /predict, /compare)
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── notebooks/
│   ├── 01_nettoyage.ipynb
│   ├── 02_visualisation.ipynb
│   ├── 03_correlation.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_modele_multivarie.ipynb
├── data/                   # voir data/README.md
│   ├── raw/                # données brutes (non versionnées)
│   ├── processed/          # co2_clean.csv
│   └── external/           # world.geojson
├── reports/figures/        # figures générées
├── requirements.txt            # application
└── requirements-notebooks.txt  # application + notebooks
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
python app/app.py
```

Puis ouvrir <http://127.0.0.1:5000>.

> Les modèles sont entraînés et évalués pour chaque pays au démarrage : comptez environ une minute avant que le serveur ne soit disponible.
> Pour le mode debug : `FLASK_DEBUG=1 python app/app.py`.

## Lancer les notebooks

```bash
pip install -r requirements-notebooks.txt
jupyter notebook notebooks/
```

Les notebooks s'exécutent dans l'ordre (01 → 05) depuis le dossier `notebooks/` ; les chemins sont relatifs à ce dossier. Le notebook 01 nécessite les données brutes (voir `data/README.md`).

## Source des données

[Our World in Data — CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data) (licence CC BY).
