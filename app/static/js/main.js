let chartA = null;
let chartB = null;
let chartCompare = null;

let currentModelB = null;
let currentScenario = null;

const SCENARIOS = window.SCENARIOS;


// =====================================================
// PRÉDICTION
// =====================================================

async function predictCO2() {

    const country =
        document.getElementById("country").value;

    const year =
        document.getElementById("year").value;

    const errorBox =
        document.getElementById("error");

    const successBox =
        document.getElementById("success");

    errorBox.style.display = "none";
    successBox.style.display = "none";


    if (!country) {

        errorBox.innerText =
            "Veuillez sélectionner un pays.";

        errorBox.style.display = "block";

        return;
    }


    if (!year || year < 2000 || year > 2040) {

        errorBox.innerText =
            "Veuillez choisir une année entre 2000 et 2040.";

        errorBox.style.display = "block";

        return;
    }


    try {

        const response =
            await fetch(
                `/predict?country=${encodeURIComponent(country)}&year=${year}`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Erreur pendant la prédiction."
            );
        }


        displayResults(data);


        successBox.innerText =
            "Prédiction calculée avec succès.";

        successBox.style.display = "block";

    }

    catch (error) {

        errorBox.innerText =
            error.message;

        errorBox.style.display =
            "block";
    }
}


// =====================================================
// AFFICHAGE DES RÉSULTATS
// =====================================================

function displayResults(data) {

    document.getElementById(
        "results"
    ).style.display = "block";


    document.getElementById(
        "resultCountry"
    ).innerText = data.country;


    document.getElementById(
        "resultYear"
    ).innerText = data.year;


    // =================================================
    // MODÈLE A
    // =================================================

    if (
        data.model_a &&
        data.model_a.available
    ) {

        const modelA =
            data.model_a;


        document.getElementById(
            "modelAPrediction"
        ).innerText =
            modelA.prediction + " t";


        document.getElementById(
            "modelAName"
        ).innerText =
            modelA.model_name;


        document.getElementById(
            "modelALastValue"
        ).innerText =
            modelA.last_value + " t";


        document.getElementById(
            "modelAEvolution"
        ).innerText =
            modelA.evolution + " %";


        drawChartA(modelA);

    }

    else {

        document.getElementById(
            "modelAPrediction"
        ).innerText = "Non disponible";


        document.getElementById(
            "modelAName"
        ).innerText = "-";


        document.getElementById(
            "modelALastValue"
        ).innerText = "-";


        document.getElementById(
            "modelAEvolution"
        ).innerText = "-";
    }


    // =================================================
    // MODÈLE B
    // =================================================

    if (
        data.model_b &&
        data.model_b.available
    ) {

        const modelB =
            data.model_b;


        document.getElementById(
            "modelBLastYear"
        ).innerText =
            modelB.last_year;


        document.getElementById(
            "modelBLastValue"
        ).innerText =
            modelB.last_value + " t";


        currentModelB = modelB;

        document.getElementById(
            "scenarioTabs"
        ).style.display = "flex";

        // Garde le scénario déjà choisi si possible,
        // sinon prend le premier de la liste.
        const scenarioToShow =
            (currentScenario && modelB.scenarios[currentScenario])
                ? currentScenario
                : SCENARIOS[0];

        selectScenario(scenarioToShow);

    }

    else {

        currentModelB = null;

        document.getElementById(
            "modelBLastYear"
        ).innerText = "Non disponible";


        document.getElementById(
            "modelBLastValue"
        ).innerText = "-";


        document.getElementById(
            "scenarioTabs"
        ).style.display = "none";


        document.getElementById(
            "scenarioContainer"
        ).innerHTML =
            "<p>Modèle B non disponible pour ce pays.</p>";


        if (chartB) {
            chartB.destroy();
            chartB = null;
        }
    }
}


// =====================================================
// SÉLECTION D'UN SCÉNARIO
// =====================================================

function selectScenario(name) {

    currentScenario = name;


    // Met en surbrillance l'onglet actif
    document
        .querySelectorAll(".scenario-tab")
        .forEach(function(tab) {

            tab.classList.toggle(
                "active",
                tab.dataset.scenario === name
            );
        });


    if (!currentModelB) {
        return;
    }


    displayScenarioDetail(
        name,
        currentModelB.scenarios[name]
    );


    drawChartB(
        currentModelB,
        name
    );
}


// =====================================================
// DÉTAIL DU SCÉNARIO CHOISI
// =====================================================

function displayScenarioDetail(name, scenario) {

    const container =
        document.getElementById(
            "scenarioContainer"
        );


    if (!scenario) {

        container.innerHTML =
            "<p>Ce scénario n'est pas disponible pour ce pays.</p>";

        return;
    }


    const value =
        scenario.prediction_at_year;


    const evolution =
        scenario.evolution;


    container.innerHTML = `

        <div class="scenario">

            <h3>${name}</h3>

            <div class="scenario-value">

                ${
                    value !== null
                    ? value + " t"
                    : "N/A"
                }

            </div>

            <p>

                Évolution par rapport à
                la dernière année observée :
                <strong>${
                    evolution !== null
                    ? evolution + " %"
                    : "N/A"
                }</strong>

            </p>

        </div>

    `;
}


// =====================================================
// GRAPHIQUE MODÈLE A
// =====================================================

function drawChartA(modelA) {

    const ctx =
        document
        .getElementById("chartA")
        .getContext("2d");


    if (chartA) {

        chartA.destroy();

    }


    const historicalYears =
        modelA.historical_years || [];


    const historicalValues =
        modelA.historical_values || [];


    const futureYears =
        modelA.future_years || [];


    const futureValues =
        modelA.future_predictions || [];


    chartA =
        new Chart(
            ctx,
            {

                type: "line",

                data: {

                    labels:
                        [
                            ...historicalYears,
                            ...futureYears
                        ],

                    datasets: [

                        {

                            label:
                                "Historique",

                            data:
                                [
                                    ...historicalValues,
                                    ...Array(
                                        futureYears.length
                                    ).fill(null)
                                ],

                            borderWidth: 3,

                            tension: 0.3

                        },

                        {

                            label:
                                "Projection",

                            data:
                                [
                                    ...Array(
                                        historicalYears.length
                                    ).fill(null),
                                    ...futureValues
                                ],

                            borderWidth: 3,

                            borderDash:
                                [6, 6],

                            tension: 0.3

                        }

                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    plugins: {

                        legend: {
                            display: true
                        }

                    },

                    scales: {

                        x: {

                            title: {

                                display: true,

                                text: "Année"

                            }

                        },

                        y: {

                            title: {

                                display: true,

                                text:
                                    "CO₂ par habitant"

                            }

                        }

                    }

                }

            }
        );
}


// =====================================================
// GRAPHIQUE MODÈLE B
// =====================================================

function drawChartB(modelB, scenarioName) {

    const ctx =
        document
        .getElementById("chartB")
        .getContext("2d");


    if (chartB) {

        chartB.destroy();

    }


    const scenario =
        modelB.scenarios[scenarioName];


    if (!scenario) {
        return;
    }


    // Historique (aligné sur les années historiques)
    const historicalYears =
        modelB.historical_years || [];

    const historicalValues =
        modelB.historical_values || [];


    // Projection du scénario choisi
    const futureYears =
        scenario.years || [];

    const futureValues =
        scenario.values || [];


    const allYears = [
        ...historicalYears,
        ...futureYears
    ];


    chartB =
        new Chart(
            ctx,
            {

                type: "line",

                data: {

                    labels:
                        allYears,

                    datasets: [

                        {

                            label:
                                "Historique",

                            data:
                                [
                                    ...historicalValues,
                                    ...Array(
                                        futureYears.length
                                    ).fill(null)
                                ],

                            borderColor:
                                "#0b1d3a",

                            backgroundColor:
                                "#0b1d3a",

                            borderWidth: 3,

                            tension: 0.3

                        },

                        {

                            label:
                                scenarioName,

                            data:
                                [
                                    ...Array(
                                        historicalYears.length
                                    ).fill(null),
                                    ...futureValues
                                ],

                            borderColor:
                                "#3b82f6",

                            backgroundColor:
                                "#3b82f6",

                            borderWidth: 3,

                            borderDash:
                                [6, 6],

                            tension: 0.3

                        }

                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    plugins: {

                        legend: {
                            display: true
                        }

                    },

                    scales: {

                        x: {

                            title: {

                                display: true,

                                text: "Année"

                            }

                        },

                        y: {

                            title: {

                                display: true,

                                text:
                                    "CO₂ par habitant"

                            }

                        }

                    }

                }

            }
        );
}


// =====================================================
// COMPARAISON ENTRE PAYS
// =====================================================

function filterCompareCountries() {

    const search =
        document
        .getElementById("compareSearch")
        .value
        .trim()
        .toLowerCase();


    document
        .querySelectorAll(
            "#compareChecklist label"
        )
        .forEach(function(label) {

            const name =
                label.dataset.name || "";

            label.style.display =
                name.includes(search)
                ? "flex"
                : "none";
        });
}


function updateCompareSelection() {

    const checked =
        Array.from(
            document.querySelectorAll(
                "#compareChecklist input:checked"
            )
        );


    const chipList =
        document.getElementById(
            "compareChips"
        );


    chipList.innerHTML = "";


    checked.forEach(function(box) {

        const chip =
            document.createElement("span");

        chip.className = "chip";

        chip.innerText = box.value;

        chipList.appendChild(chip);
    });
}


function clearCompareSelection() {

    document
        .querySelectorAll(
            "#compareChecklist input:checked"
        )
        .forEach(function(box) {
            box.checked = false;
        });


    updateCompareSelection();


    document.getElementById(
        "compareResults"
    ).classList.add("hidden");


    document.getElementById(
        "compareError"
    ).style.display = "none";
}


async function compareCountries() {

    const checked =
        Array.from(
            document.querySelectorAll(
                "#compareChecklist input:checked"
            )
        ).map(function(box) {
            return box.value;
        });


    const year =
        document.getElementById(
            "compareYear"
        ).value;


    const scenario =
        document.getElementById(
            "compareScenario"
        ).value;


    const errorBox =
        document.getElementById(
            "compareError"
        );

    errorBox.style.display = "none";


    if (checked.length < 2) {

        errorBox.innerText =
            "Sélectionnez au moins deux pays à comparer.";

        errorBox.style.display = "block";

        return;
    }


    if (!year || year < 2000 || year > 2040) {

        errorBox.innerText =
            "Veuillez choisir une année entre 2000 et 2040.";

        errorBox.style.display = "block";

        return;
    }


    try {

        const params =
            new URLSearchParams({

                countries:
                    checked.join(","),

                year: year,

                scenario: scenario
            });


        const response =
            await fetch(
                `/compare?${params.toString()}`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Erreur pendant la comparaison."
            );
        }


        displayCompareResults(data);

    }

    catch (error) {

        errorBox.innerText =
            error.message;

        errorBox.style.display =
            "block";
    }
}


function displayCompareResults(data) {

    document.getElementById(
        "compareResults"
    ).classList.remove("hidden");


    const tbody =
        document.getElementById(
            "compareTableBody"
        );

    tbody.innerHTML = "";


    data.results.forEach(function(row) {

        const tr =
            document.createElement("tr");

        tr.innerHTML = `
            <td>${row.country}</td>
            <td>${row.prediction} t</td>
            <td>${row.model}</td>
        `;

        tbody.appendChild(tr);
    });


    const ctx =
        document
        .getElementById("chartCompare")
        .getContext("2d");


    if (chartCompare) {
        chartCompare.destroy();
    }


    chartCompare =
        new Chart(
            ctx,
            {

                type: "bar",

                data: {

                    labels:
                        data.results.map(
                            function(r) {
                                return r.country;
                            }
                        ),

                    datasets: [

                        {

                            label:
                                `CO₂/habitant en ${data.year} (${data.scenario})`,

                            data:
                                data.results.map(
                                    function(r) {
                                        return r.prediction;
                                    }
                                ),

                            backgroundColor:
                                "#3b82f6",

                            borderColor:
                                "#172554",

                            borderWidth: 1.5

                        }

                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    plugins: {

                        legend: {
                            display: true
                        }

                    },

                    scales: {

                        y: {

                            title: {

                                display: true,

                                text:
                                    "CO₂ par habitant (t)"

                            }

                        }

                    }

                }

            }
        );
}
