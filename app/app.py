import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)


# =========================================================
# APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

TARGET = "co2_per_capita"

TRAIN_END = 2015
TEST_END = 2024
FUTURE_END = 2040

MIN_POINTS = 15
MIN_OBS_MULTI = 20


# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "processed", "co2_clean.csv")
)


print("=" * 60)
print("CHARGEMENT DES DONNÉES")
print("=" * 60)
print("Fichier :", DATA_PATH)


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Fichier introuvable : {DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)


print("Nombre de lignes :", len(df))


# =========================================================
# VÉRIFICATION DES COLONNES
# =========================================================

required_columns = [
    "country",
    "year",
    "co2_per_capita",
    "gdp",
    "population",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "co2"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "Colonnes manquantes dans le CSV : "
        + ", ".join(missing_columns)
    )


# =========================================================
# NETTOYAGE
# =========================================================

NUMERIC_COLUMNS = [c for c in required_columns if c != "country"]

for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.dropna(
    subset=[
        "country",
        "year"
    ]
).copy()


df["year"] = df["year"].astype(int)


print(
    "Nombre de pays :",
    df["country"].nunique()
)


# =========================================================
# MODÈLE A — ANNÉE UNIQUEMENT
# =========================================================

_point_counts = (

    df.dropna(
        subset=[TARGET]
    )

    .groupby("country")["year"]

    .count()
)


COUNTRIES_SIMPLE = sorted(

    _point_counts[
        _point_counts >= MIN_POINTS
    ]

    .index

    .tolist()
)


print()
print("=" * 60)
print("MODÈLE A")
print("=" * 60)

print(
    "Pays disponibles :",
    len(COUNTRIES_SIMPLE)
)


# =========================================================
# DONNÉES PAR PAYS
# =========================================================

country_data = {}


for country in COUNTRIES_SIMPLE:

    sub = (

        df[
            df["country"] == country
        ]

        [
            [
                "year",
                TARGET
            ]
        ]

        .dropna(
            subset=[TARGET]
        )

        .sort_values("year")

        .drop_duplicates(
            subset=["year"],
            keep="last"
        )

    )

    country_data[country] = sub


# =========================================================
# TRAIN / TEST MODÈLE A
# =========================================================

def split_train_test(sub):

    train = sub[
        sub["year"] <= TRAIN_END
    ].copy()

    test = sub[
        (sub["year"] > TRAIN_END)
        &
        (sub["year"] <= TEST_END)
    ].copy()

    return train, test


# =========================================================
# MODÈLES A
# =========================================================

def make_simple_models():

    return {

        "Régression (deg 2)":
            make_pipeline(

                PolynomialFeatures(
                    degree=2
                ),

                LinearRegression()
            ),

        "Random Forest":
            RandomForestRegressor(

                n_estimators=100,

                max_depth=4,

                random_state=42,

                n_jobs=-1
            )
    }


# =========================================================
# ÉVALUATION MODÈLE A
# =========================================================

print()
print("=" * 60)
print("ÉVALUATION DU MODÈLE A")
print("=" * 60)


simple_results = []


for i, country in enumerate(
    COUNTRIES_SIMPLE,
    start=1
):

    sub = country_data[country]

    train, test = split_train_test(
        sub
    )


    if (
        len(train) < 5
        or
        len(test) == 0
    ):
        continue


    print(
        f"[{i}/{len(COUNTRIES_SIMPLE)}] {country}"
    )


    for model_name, model in (
        make_simple_models().items()
    ):

        try:

            model.fit(
                train[["year"]],
                train[TARGET]
            )


            prediction = model.predict(
                test[["year"]]
            )


            rmse = np.sqrt(
                mean_squared_error(
                    test[TARGET],
                    prediction
                )
            )


            mae = mean_absolute_error(
                test[TARGET],
                prediction
            )


            # Éviter les problèmes avec les valeurs nulles
            non_zero_mask = (
                test[TARGET] != 0
            )


            if non_zero_mask.any():

                mape = (

                    mean_absolute_percentage_error(

                        test.loc[
                            non_zero_mask,
                            TARGET
                        ],

                        prediction[
                            non_zero_mask
                        ]
                    )

                    * 100
                )

            else:

                mape = 0


            simple_results.append({

                "country":
                    country,

                "model":
                    model_name,

                "RMSE":
                    round(
                        float(rmse),
                        3
                    ),

                "MAE":
                    round(
                        float(mae),
                        3
                    ),

                "MAPE":
                    round(
                        float(mape),
                        1
                    )
            })


        except Exception as e:

            print(
                f"Erreur modèle {model_name} "
                f"pour {country} : {e}"
            )


simple_metrics_df = pd.DataFrame(
    simple_results
)


# =========================================================
# MEILLEUR MODÈLE PAR PAYS
# =========================================================

if not simple_metrics_df.empty:

    best_indices = (

        simple_metrics_df

        .groupby("country")["RMSE"]

        .idxmin()
    )


    simple_best = (

        simple_metrics_df

        .loc[best_indices]

        .set_index("country")
    )

else:

    simple_best = pd.DataFrame(
        columns=[
            "model",
            "RMSE",
            "MAE",
            "MAPE"
        ]
    )

    simple_best.index.name = "country"


# =========================================================
# PROJECTIONS MODÈLE A
# =========================================================

print()
print("=" * 60)
print("PROJECTIONS MODÈLE A")
print("=" * 60)


simple_projections = {}


for i, country in enumerate(
    COUNTRIES_SIMPLE,
    start=1
):

    sub = country_data[country]


    if country not in simple_best.index:
        continue


    best_model_name = (

        simple_best

        .loc[
            country,
            "model"
        ]
    )


    model = (

        make_simple_models()

        [best_model_name]
    )


    print(
        f"[{i}/{len(COUNTRIES_SIMPLE)}] "
        f"{country} -> {best_model_name}"
    )


    # -----------------------------------------------------
    # Entraînement final
    # -----------------------------------------------------

    model.fit(
        sub[["year"]],
        sub[TARGET]
    )


    # -----------------------------------------------------
    # Résidus
    # -----------------------------------------------------

    fitted_values = model.predict(
        sub[["year"]]
    )


    residuals = (
        sub[TARGET].values
        - fitted_values
    )


    residual_std = (
        np.std(residuals)
        if len(residuals) > 1
        else 0
    )


    # -----------------------------------------------------
    # Années futures
    # -----------------------------------------------------

    last_year = int(
        sub["year"].max()
    )


    future_years = np.arange(

        last_year + 1,

        FUTURE_END + 1
    )


    if len(future_years) > 0:

        future_predictions = (
            model.predict(
                future_years.reshape(-1, 1)
            )
        )

        # CO2 ne peut pas être négatif
        future_predictions = np.maximum(
            future_predictions,
            0
        )

    else:

        future_predictions = np.array([])


    # -----------------------------------------------------
    # Sauvegarde
    # -----------------------------------------------------

    simple_projections[country] = {

        "model":
            model,

        "model_name":
            best_model_name,

        "historical_years":
            sub["year"].astype(int).tolist(),

        "historical_values":
            sub[TARGET].astype(float).tolist(),

        "future_years":
            future_years.astype(int).tolist(),

        "future_predictions":
            future_predictions.astype(float).tolist(),

        "residual_std":
            float(residual_std),

        "last_year":
            last_year,

        "last_value":
            float(
                sub.iloc[-1][TARGET]
            )
    }


# =========================================================
# MODÈLE B — MULTIVARIÉ
# =========================================================

print()
print("=" * 60)
print("PRÉPARATION DU MODÈLE B")
print("=" * 60)


# =========================================================
# PIB PAR HABITANT
# =========================================================

df["gdp_per_capita"] = np.where(

    df["population"] > 0,

    df["gdp"] / df["population"],

    np.nan
)


# =========================================================
# PART FOSSILE
# =========================================================

fossil_numerator = (

    df["coal_co2"]

    + df["oil_co2"]

    + df["gas_co2"]
)


df["fossil_share"] = np.where(

    df["co2"] > 0,

    fossil_numerator / df["co2"],

    np.nan
)


df["fossil_share"] = (

    df["fossil_share"]

    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


FEATURES_MULTI = [

    "year",

    "population",

    "gdp_per_capita",

    "fossil_share"
]


MULTI_VARS = (

    FEATURES_MULTI

    + [TARGET]
)


# =========================================================
# PAYS DU MODÈLE B
# =========================================================

_multi_counts = (

    df

    .groupby("country")[MULTI_VARS]

    .apply(

        lambda x:

        x.notna()

        .all(axis=1)

        .sum()
    )
)


COUNTRIES_MULTI = sorted(

    _multi_counts[

        _multi_counts
        >= MIN_OBS_MULTI
    ]

    .index

    .tolist()
)


print(
    "Pays modèle B :",
    len(COUNTRIES_MULTI)
)


# =========================================================
# DONNÉES MODÈLE B
# =========================================================

df_model = (

    df[
        df["country"].isin(
            COUNTRIES_MULTI
        )
    ]

    .dropna(
        subset=MULTI_VARS
    )

    .copy()
)


# =========================================================
# TRAIN / TEST MODÈLE B
# =========================================================

multi_train = df_model[

    df_model["year"]
    <= TRAIN_END
].copy()


multi_test = df_model[

    (df_model["year"] > TRAIN_END)

    &

    (df_model["year"] <= TEST_END)
].copy()


X_train = multi_train[
    FEATURES_MULTI
]


y_train = multi_train[
    TARGET
]


X_test = multi_test[
    FEATURES_MULTI
]


y_test = multi_test[
    TARGET
]


# =========================================================
# MODÈLE B ÉVALUATION
# =========================================================

print()
print("Entraînement du modèle multivarié...")


multi_eval_model = RandomForestRegressor(

    n_estimators=100,

    max_depth=10,

    random_state=42,

    n_jobs=-1
)


if (
    len(X_train) == 0
    or
    len(X_test) == 0
):

    print(
        "Attention : données train/test "
        "insuffisantes pour le modèle B."
    )

    multi_metrics = {

        "MAE": 0,

        "RMSE": 0,

        "R2": 0,

        "n_countries":
            int(
                df_model["country"].nunique()
            ),

        "n_train":
            int(len(multi_train)),

        "n_test":
            int(len(multi_test))
    }


else:

    multi_eval_model.fit(
        X_train,
        y_train
    )


    y_pred = (
        multi_eval_model.predict(
            X_test
        )
    )


    multi_metrics = {

        "MAE":
            round(
                float(
                    mean_absolute_error(
                        y_test,
                        y_pred
                    )
                ),
                3
            ),

        "RMSE":
            round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            y_pred
                        )
                    )
                ),
                3
            ),

        "R2":
            round(
                float(
                    r2_score(
                        y_test,
                        y_pred
                    )
                ),
                3
            ),

        "n_countries":
            int(
                df_model[
                    "country"
                ].nunique()
            ),

        "n_train":
            int(len(multi_train)),

        "n_test":
            int(len(multi_test))
    }


# =========================================================
# MODÈLE B FINAL
# =========================================================

print(
    "Entraînement du modèle final..."
)


multi_final_model = RandomForestRegressor(

    n_estimators=100,

    max_depth=10,

    random_state=42,

    n_jobs=-1
)


if len(df_model) > 0:

    multi_final_model.fit(

        df_model[
            FEATURES_MULTI
        ],

        df_model[
            TARGET
        ]
    )


# =========================================================
# IMPORTANCE DES VARIABLES
# =========================================================

if len(df_model) > 0:

    feature_importance_df = (

        pd.DataFrame({

            "Variable":
                FEATURES_MULTI,

            "Importance":
                multi_final_model
                .feature_importances_
        })

        .sort_values(
            "Importance",
            ascending=False
        )
    )

else:

    feature_importance_df = pd.DataFrame({

        "Variable":
            FEATURES_MULTI,

        "Importance":
            [0.0] * len(FEATURES_MULTI)
    })


feature_importance_list = [

    {

        "variable":
            row.Variable,

        "importance":
            round(
                float(
                    row.Importance
                ),
                3
            )
    }

    for row
    in feature_importance_df.itertuples()
]


# =========================================================
# SCÉNARIOS
# =========================================================

SCENARIOS = [

    "Continuité",

    "Transition énergétique",

    "Forte croissance"
]


_future_rows = []


# =========================================================
# CRÉATION DES SCÉNARIOS
# =========================================================

for country in COUNTRIES_MULTI:

    country_df = (

        df_model[
            df_model["country"]
            == country
        ]

        .sort_values("year")
    )


    if country_df.empty:
        continue


    last = country_df.iloc[-1]


    population_last = float(
        last["population"]
    )


    gdp_last = float(
        last["gdp_per_capita"]
    )


    fossil_last = float(
        last["fossil_share"]
    )


    last_year = int(
        last["year"]
    )


    # -----------------------------------------------------
    # Croissance population
    # -----------------------------------------------------

    pop_changes = (

        country_df[
            "population"
        ]

        .pct_change()

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .dropna()
    )


    if len(pop_changes) > 0:

        pop_growth = float(
            pop_changes.mean()
        )

    else:

        pop_growth = 0.0


    # -----------------------------------------------------
    # Croissance PIB
    # -----------------------------------------------------

    gdp_changes = (

        country_df[
            "gdp_per_capita"
        ]

        .pct_change()

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .dropna()
    )


    if len(gdp_changes) > 0:

        gdp_growth = float(
            gdp_changes.mean()
        )

    else:

        gdp_growth = 0.0


    # Limites réalistes

    pop_growth = np.clip(
        pop_growth,
        -0.02,
        0.05
    )


    gdp_growth = np.clip(
        gdp_growth,
        -0.05,
        0.08
    )


    fossil_last = np.clip(
        fossil_last,
        0,
        1
    )


    # =====================================================
    # ANNÉES FUTURES
    # =====================================================

    for year in range(

        last_year + 1,

        FUTURE_END + 1
    ):

        n = year - last_year


        # -------------------------------------------------
        # Population
        # -------------------------------------------------

        population_future = (

            population_last

            *

            (
                1 + pop_growth
            ) ** n
        )


        # =================================================
        # CONTINUITÉ
        # =================================================

        gdp_continuity = (

            gdp_last

            *

            (
                1 + gdp_growth
            ) ** n
        )


        fossil_continuity = (
            fossil_last
        )


        # =================================================
        # TRANSITION ÉNERGÉTIQUE
        # =================================================

        gdp_transition = (

            gdp_last

            *

            (
                1 + gdp_growth
            ) ** n
        )


        fossil_transition = max(

            fossil_last

            *

            (
                0.98 ** n
            ),

            0
        )


        # =================================================
        # FORTE CROISSANCE
        # =================================================

        gdp_high = (

            gdp_last

            *

            (
                1
                + gdp_growth
                + 0.01
            ) ** n
        )


        fossil_high = (
            fossil_last
        )


        # =================================================
        # AJOUT DES SCÉNARIOS
        # =================================================

        _future_rows.extend([

            [

                country,

                year,

                "Continuité",

                population_future,

                gdp_continuity,

                fossil_continuity
            ],

            [

                country,

                year,

                "Transition énergétique",

                population_future,

                gdp_transition,

                fossil_transition
            ],

            [

                country,

                year,

                "Forte croissance",

                population_future,

                gdp_high,

                fossil_high
            ]

        ])


# =========================================================
# DATAFRAME FUTUR
# =========================================================

future_scenarios = pd.DataFrame(

    _future_rows,

    columns=[

        "country",

        "year",

        "scenario",

        "population",

        "gdp_per_capita",

        "fossil_share"
    ]
)


# =========================================================
# PRÉDICTIONS FUTURES
# =========================================================

if (
    len(future_scenarios) > 0
    and
    len(df_model) > 0
):

    future_scenarios[
        "predicted_co2_per_capita"
    ] = (

        multi_final_model.predict(

            future_scenarios[
                FEATURES_MULTI
            ]
        )
    )


    future_scenarios[
        "predicted_co2_per_capita"
    ] = np.maximum(

        future_scenarios[
            "predicted_co2_per_capita"
        ],

        0
    )

else:

    future_scenarios[
        "predicted_co2_per_capita"
    ] = []


# =========================================================
# HISTORIQUE MODÈLE B
# =========================================================

multi_history = {}


for country in COUNTRIES_MULTI:

    hist = (

        df_model[
            df_model["country"]
            == country
        ]

        .sort_values("year")

        [
            [
                "year",
                TARGET
            ]
        ]
    )


    multi_history[country] = hist


# =========================================================
# TOUS LES PAYS
# =========================================================

ALL_COUNTRIES = sorted(

    set(COUNTRIES_SIMPLE)

    |

    set(COUNTRIES_MULTI)
)


# =========================================================
# PAGE PRINCIPALE
# =========================================================

@app.route("/")
def index():

    return render_template(

        "index.html",

        countries=ALL_COUNTRIES,

        countries_simple=
            COUNTRIES_SIMPLE,

        countries_multi=
            COUNTRIES_MULTI,

        scenarios=SCENARIOS,

        best_models=(

            simple_best

            .reset_index()

            .to_dict(
                orient="records"
            )
        ),

        metrics=(

            simple_metrics_df

            .to_dict(
                orient="records"
            )
        ),

        multi_metrics=
            multi_metrics,

        feature_importance=
            feature_importance_list
    )


# =========================================================
# API /predict
# =========================================================

@app.route(
    "/predict",
    methods=["GET"]
)
def predict():

    try:

        country = request.args.get(
            "country"
        )


        if not country:

            return jsonify({

                "error":
                    "Aucun pays sélectionné."
            }), 400


        year = int(

            request.args.get(
                "year",
                2030
            )
        )


        if year < 1900:

            return jsonify({

                "error":
                    "Année invalide."
            }), 400


        result = {

            "country":
                country,

            "year":
                year
        }


        # =================================================
        # MODÈLE A
        # =================================================

        if country in simple_projections:

            proj = (

                simple_projections[
                    country
                ]
            )


            model = proj["model"]


            pred_a = float(

                model.predict(

                    np.array([
                        [year]
                    ])
                )[0]
            )


            pred_a = max(
                pred_a,
                0
            )


            last_value = (

                proj[
                    "last_value"
                ]
            )


            if last_value != 0:

                evolution_a = (

                    (
                        pred_a
                        - last_value
                    )

                    /

                    last_value

                    *

                    100
                )

            else:

                evolution_a = 0


            result["model_a"] = {

                "available":
                    True,

                "model_name":
                    proj[
                        "model_name"
                    ],

                "prediction":
                    round(
                        pred_a,
                        2
                    ),

                "evolution":
                    round(
                        evolution_a,
                        1
                    ),

                "last_year":
                    proj[
                        "last_year"
                    ],

                "last_value":
                    round(
                        last_value,
                        2
                    ),

                "historical_years":
                    proj[
                        "historical_years"
                    ],

                "historical_values":
                    proj[
                        "historical_values"
                    ],

                "future_years":
                    proj[
                        "future_years"
                    ],

                "future_predictions":
                    proj[
                        "future_predictions"
                    ]
            }


        else:

            result["model_a"] = {

                "available":
                    False
            }


        # =================================================
        # MODÈLE B
        # =================================================

        if country in COUNTRIES_MULTI:

            hist = (

                multi_history[
                    country
                ]
            )


            full_future = (

                future_scenarios[

                    future_scenarios[
                        "country"
                    ]
                    == country
                ]
            )


            last_hist_value = float(

                hist[
                    TARGET
                ].iloc[-1]
            )


            scenarios_payload = {}


            for scenario in SCENARIOS:

                s = (

                    full_future[

                        full_future[
                            "scenario"
                        ]
                        == scenario
                    ]

                    .sort_values(
                        "year"
                    )
                )


                at_year = s[

                    s["year"]
                    == year
                ]


                if len(at_year) > 0:

                    pred_b = float(

                        at_year[
                            "predicted_co2_per_capita"
                        ].iloc[0]
                    )

                else:

                    pred_b = None


                if (

                    pred_b is not None

                    and

                    last_hist_value != 0
                ):

                    evolution_b = round(

                        (
                            (
                                pred_b
                                - last_hist_value
                            )

                            /

                            last_hist_value

                            *

                            100
                        ),

                        1
                    )

                else:

                    evolution_b = None


                scenarios_payload[
                    scenario
                ] = {

                    "prediction_at_year":

                        (
                            round(
                                pred_b,
                                2
                            )

                            if pred_b
                            is not None

                            else None
                        ),

                    "evolution":
                        evolution_b,

                    "years":
                        s[
                            "year"
                        ].astype(int).tolist(),

                    "values":

                        [

                            round(
                                float(v),
                                3
                            )

                            for v

                            in s[
                                "predicted_co2_per_capita"
                            ].tolist()
                        ]
                }


            result["model_b"] = {

                "available":
                    True,

                "historical_years":

                    hist[
                        "year"
                    ].astype(int).tolist(),

                "historical_values":

                    hist[
                        TARGET
                    ].astype(float).tolist(),

                "last_year":

                    int(
                        hist[
                            "year"
                        ].iloc[-1]
                    ),

                "last_value":

                    round(
                        last_hist_value,
                        2
                    ),

                "scenarios":
                    scenarios_payload
            }


        else:

            result["model_b"] = {

                "available":
                    False
            }


        # =================================================
        # VÉRIFICATION
        # =================================================

        if (

            not result[
                "model_a"
            ][
                "available"
            ]

            and

            not result[
                "model_b"
            ][
                "available"
            ]
        ):

            return jsonify({

                "error":
                    "Pays non disponible dans aucun des deux modèles."
            }), 400


        return jsonify(result)


    except ValueError:

        return jsonify({

            "error":
                "L'année doit être un nombre entier."
        }), 400


    except Exception as e:

        print(
            "ERREUR /predict :",
            str(e)
        )


        return jsonify({

            "error":
                str(e)
        }), 500


# =========================================================
# COMPARAISON MULTI-PAYS
# =========================================================

@app.route(
    "/compare",
    methods=["GET"]
)
def compare():

    try:

        countries_param = request.args.get(
            "countries",
            ""
        )


        year = int(

            request.args.get(
                "year",
                2030
            )
        )


        scenario = request.args.get(
            "scenario",
            "Continuité"
        )


        if scenario not in SCENARIOS:

            scenario = "Continuité"


        if not countries_param:

            return jsonify({

                "error":
                    "Aucun pays sélectionné."
            }), 400


        countries = [

            c.strip()

            for c
            in countries_param.split(",")

            if c.strip()
        ]


        results = []
        unavailable = []


        for country in countries:

            # =============================================
            # PRIORITÉ AU MODÈLE B (scénario choisi)
            # =============================================

            if country in COUNTRIES_MULTI:

                row = future_scenarios[

                    (future_scenarios["country"] == country)
                    &
                    (future_scenarios["scenario"] == scenario)
                    &
                    (future_scenarios["year"] == year)
                ]


                if len(row) > 0:

                    prediction = float(

                        row[
                            "predicted_co2_per_capita"
                        ].iloc[0]
                    )


                    results.append({

                        "country":
                            country,

                        "prediction":
                            round(
                                max(prediction, 0),
                                2
                            ),

                        "model":
                            f"Modèle B ({scenario})",

                        "source":
                            "model_b"
                    })

                    continue


            # =============================================
            # REPLI SUR LE MODÈLE A
            # =============================================

            if country in simple_projections:

                proj = (
                    simple_projections[
                        country
                    ]
                )


                model = proj["model"]


                prediction = model.predict(

                    np.array([
                        [year]
                    ])
                )[0]


                prediction = max(
                    float(prediction),
                    0
                )


                results.append({

                    "country":
                        country,

                    "prediction":
                        round(
                            prediction,
                            2
                        ),

                    "model":
                        proj[
                            "model_name"
                        ],

                    "source":
                        "model_a"
                })

                continue


            unavailable.append(country)


        if not results:

            return jsonify({

                "error":
                    "Aucun des pays sélectionnés "
                    "n'est disponible pour cette comparaison."
            }), 400


        return jsonify({

            "year":
                year,

            "scenario":
                scenario,

            "results":
                results,

            "unavailable":
                unavailable
        })


    except ValueError:

        return jsonify({

            "error":
                "L'année doit être un nombre entier."
        }), 400


    except Exception as e:

        print(
            "ERREUR /compare :",
            str(e)
        )


        return jsonify({

            "error":
                str(e)
        }), 500


# =========================================================
# LANCEMENT
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("APPLICATION FLASK")
    print("=" * 60)

    print(
        f"Pays modèle A : "
        f"{len(COUNTRIES_SIMPLE)}"
    )

    print(
        f"Pays modèle B : "
        f"{len(COUNTRIES_MULTI)}"
    )

    print(
        f"Total pays : "
        f"{len(ALL_COUNTRIES)}"
    )

    print("=" * 60)

    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
