# Données

| Dossier | Contenu | Produit par |
|---|---|---|
| `raw/` | Données brutes `owid-co2-data.csv` (non versionné, voir ci-dessous) | téléchargement |
| `processed/` | `co2_clean.csv` : données nettoyées, utilisées par l'application et les notebooks 02 à 05 | `notebooks/01_nettoyage.ipynb` |
| `external/` | `world.geojson` : fond de carte pour les cartes du notebook 02 | source externe |

## Récupérer les données brutes

Le fichier brut provient du jeu de données CO₂ d'[Our World in Data](https://github.com/owid/co2-data) :

```bash
curl -o data/raw/owid-co2-data.csv https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv
```

Il n'est nécessaire que pour relancer `01_nettoyage.ipynb`. L'application fonctionne avec `processed/co2_clean.csv`, qui est versionné.

Les données OWID sont diffusées sous licence CC BY : pensez à citer la source.
