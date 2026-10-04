"""
Regresión logística para predicción de abandono de clientes (Customer Churn).

Archivo independiente. Ejecución:
python modelo.py

Con tus datos:
python modelo.py --datos "telco.csv"

También acepta .xlsx:
python modelo.py --datos "datos.xlsx" --hoja Nombre

El objetivo es Churn:
    No  = cliente permanece
    Yes = cliente abandona

El modelo utiliza las características del cliente para estimar
la probabilidad de abandono.

No se utiliza customerID como predictor.
"""

# 1. CONFIGURACIÓN: dataset, objetivo y opciones de ejecución.
RUTA_DATASET = "telco.csv"
COLUMNA_OBJETIVO = "Churn"
TAREA = "clasificacion"
MOSTRAR_GRAFICAS = True
HOJA_EXCEL = 0
COLUMNAS_ENTRADA = []

NOTA_MODELO = (
    "Problema: predicción del abandono de clientes. "
    "Clase positiva: Yes = Abandona."
)

# Bibliotecas para datos, preparación y evaluación.
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

SEMILLA = 42


# 2. DATOS: lectura, opciones y validación.
def leer_tabla(ruta, separador=None, hoja=0):
    """Lee CSV o Excel y resuelve las rutas desde la carpeta del programa."""

    ruta = Path(ruta).expanduser()

    if not ruta.is_absolute():
        ruta = Path(__file__).resolve().parent / ruta

    if not ruta.is_file():
        raise ValueError(f"No se encontró el dataset: {ruta}")

    if ruta.suffix.lower() == ".csv":
        tabla = pd.read_csv(
            ruta,
            sep=separador,
            engine="python",
            encoding="utf-8-sig"
        )

    elif ruta.suffix.lower() == ".xlsx":
        tabla = pd.read_excel(
            ruta,
            sheet_name=hoja,
            engine="openpyxl"
        )

    else:
        raise ValueError(
            "El dataset debe ser un archivo .csv o .xlsx."
        )

    if tabla.empty:
        raise ValueError("El dataset está vacío.")

    tabla.columns = tabla.columns.map(str)

    return tabla, str(ruta.resolve())


def argumentos(titulo, tareas):
    """Recoge las opciones de ejecución."""

    parser = argparse.ArgumentParser(description=titulo)

    parser.add_argument(
        "--tarea",
        choices=tareas,
        default=TAREA
    )

    parser.add_argument(
        "--datos",
        "--csv",
        "--xlsx",
        dest="datos",
        default=RUTA_DATASET,
        help="Ruta al dataset .csv o .xlsx"
    )

    parser.add_argument(
        "--objetivo",
        default=COLUMNA_OBJETIVO,
        help="Columna que se predice"
    )

    parser.add_argument(
        "--hoja",
        default=HOJA_EXCEL,
        help="Nombre o índice de hoja Excel"
    )

    parser.add_argument(
        "--separador",
        default=None,
        help="Separador CSV"
    )

    parser.add_argument(
        "--columnas",
        nargs="+",
        default=COLUMNAS_ENTRADA or None,
        help="Variables que se utilizarán como predictores"
    )

    parser.add_argument(
        "--salida",
        type=Path,
        help="Opcional: guardar resultados en esta carpeta"
    )

    visual = parser.add_mutually_exclusive_group()

    visual.add_argument(
        "--graficas",
        dest="graficas",
        action="store_true",
        help="Mostrar gráficos"
    )

    visual.add_argument(
        "--sin-graficas",
        dest="graficas",
        action="store_false",
        help="No abrir ventanas"
    )

    parser.set_defaults(graficas=MOSTRAR_GRAFICAS)

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecutar ejemplo pequeño"
    )

    args = parser.parse_args()

    if args.demo:
        args.datos = None
        args.objetivo = None
        args.columnas = None

    if args.tarea not in tareas:
        parser.error(
            f"TAREA debe ser una de: {tareas}"
        )

    args.hoja = (
        int(args.hoja)
        if str(args.hoja).isdigit()
        else args.hoja
    )

    return args


def cargar_datos(args, ejemplo="general"):
    """Carga los datos y prepara Churn como objetivo."""

    if args.datos:

        tabla, fuente = leer_tabla(
            args.datos,
            args.separador,
            args.hoja
        )

        # Detectar automáticamente si es el dataset Telco.
        args.es_telco = {
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        }.issubset(tabla.columns)

        if args.es_telco:

            tabla = tabla.copy()

            # Validar identificador.
            if (
                tabla.customerID.isna().any()
                or tabla.customerID.duplicated().any()
            ):
                raise ValueError(
                    "Cada cliente debe tener un identificador "
                    "único y no vacío."
                )

            # Limpiar variables de texto.
            for col in tabla.select_dtypes(exclude="number"):

                tabla[col] = (
                    tabla[col]
                    .astype("string")
                    .str.strip()
                    .replace("", np.nan)
                )

            # Convertir variables numéricas.
            for col in [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]:

                tabla[col] = pd.to_numeric(
                    tabla[col],
                    errors="coerce"
                ).astype(float)

                if np.isinf(tabla[col]).any():
                    raise ValueError(
                        f"Valores infinitos en {col}."
                    )

                if tabla[col].dropna().lt(0).any():
                    raise ValueError(
                        f"Valores negativos inválidos en {col}."
                    )

            # Validar variable objetivo.
            if not tabla.Churn.dropna().isin(["Yes", "No"]).all():
                raise ValueError(
                    "Churn debe contener únicamente Yes o No."
                )

            if tabla.Churn.isna().any():
                raise ValueError(
                    "Churn contiene valores vacíos."
                )

            # Convertir SeniorCitizen a categoría.
            if "SeniorCitizen" in tabla:

                if not tabla.SeniorCitizen.isin([0, 1]).all():
                    raise ValueError(
                        "SeniorCitizen debe ser 0 o 1."
                    )

                tabla["SeniorCitizen"] = (
                    tabla.SeniorCitizen
                    .map({
                        0: "No",
                        1: "Yes"
                    })
                )

            # Para este modelo el objetivo SIEMPRE es Churn.
            esperado = "Churn"

            if (
                args.objetivo is not None
                and args.objetivo != esperado
            ):
                raise ValueError(
                    "Para este modelo el objetivo debe ser Churn."
                )

            args.objetivo = esperado

            # Excluir identificador y objetivo.
            excluir = {
                "customerID",
                "Churn"
            }

            permitidas = [
                c for c in tabla.columns
                if c not in excluir
            ]

            if args.columnas is None:
                args.columnas = permitidas

            if not set(args.columnas).issubset(permitidas):

                raise ValueError(
                    "Las variables de entrada no pueden contener "
                    "customerID ni Churn."
                )

            # Guardar customerID como índice.
            tabla = tabla.set_index("customerID")

            print(
                "Dataset Telco detectado: "
                "el objetivo es predecir Churn."
            )

            print(
                "Yes = Abandona | No = Permanece"
            )

        elif args.objetivo is None:

            raise ValueError(
                "Para otro dataset indica --objetivo."
            )

        if args.objetivo not in tabla:

            raise ValueError(
                f"No existe la columna objetivo "
                f"'{args.objetivo}'. "
                f"Columnas: {list(tabla.columns)}"
            )

        if tabla[args.objetivo].isna().any():

            raise ValueError(
                "La columna objetivo contiene valores vacíos."
            )

        X = tabla.drop(
            columns=args.objetivo
        )

        y = tabla[args.objetivo]

    else:

        datos = load_wine(as_frame=True)

        X = datos.data
        y = datos.target

        fuente = (
            "DEMOSTRACIÓN: Wine de scikit-learn"
        )

    # Selección manual de variables.
    if args.columnas:

        faltantes = (
            set(args.columnas)
            - set(X.columns)
        )

        if faltantes:

            raise ValueError(
                f"Variables inexistentes: "
                f"{sorted(faltantes)}"
            )

        if len(args.columnas) != len(set(args.columnas)):

            raise ValueError(
                "No repitas nombres en las columnas."
            )

        X = X[
            args.columnas
        ].copy()

    if X.shape[1] == 0 or len(X) < 20:

        raise ValueError(
            "Se necesitan al menos 20 filas "
            "y una variable predictora."
        )

    # Validar valores infinitos.
    if (
        X.select_dtypes(include="number")
        .to_numpy()
        .size > 0
    ):

        if np.isinf(
            X.select_dtypes(include="number")
            .to_numpy()
        ).any():

            raise ValueError(
                "Hay valores infinitos."
            )

    # Convertir variables categóricas a texto.
    for columna in X.select_dtypes(
        exclude="number"
    ).columns:

        X[columna] = X[columna].map(
            lambda v: str(v)
            if pd.notna(v)
            else np.nan
        )

    # Codificación explícita de Churn.
    if args.tarea == "clasificacion":

        if args.objetivo == "Churn":

            mapa = {
                "No": 0,
                "Yes": 1
            }

            y = y.map(mapa)

            if y.isna().any():

                raise ValueError(
                    "Churn debe contener únicamente "
                    "'Yes' o 'No'."
                )

            y = y.astype(int)

            args.clases = [
                "No",
                "Yes"
            ]

            print(
                "Clases: "
                "{0: 'No = Permanece', "
                "1: 'Yes = Abandona'}"
            )

            fuente += (
                "; clase positiva = Yes (Abandona)"
            )

        else:

            from sklearn.preprocessing import LabelEncoder

            encoder = LabelEncoder()

            y = pd.Series(
                encoder.fit_transform(
                    y.astype(str)
                ),
                index=X.index,
                name=y.name
            )

            if (
                y.nunique() < 2
                or y.value_counts().min() < 8
            ):

                raise ValueError(
                    "Se requieren al menos dos clases "
                    "y ocho filas por clase."
                )

            args.clases = [
                str(c)
                for c in encoder.classes_
            ]

    return X, y, fuente


# 3. PREPARACIÓN.
def preparar(X, modelo, escalar=False):
    """Une preparación de datos y modelo."""

    numericas = list(
        X.select_dtypes(
            include="number"
        ).columns
    )

    categoricas = [
        c for c in X.columns
        if c not in numericas
    ]

    transformaciones = []

    if numericas:

        pasos = [
            (
                "imputar",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            )
        ]

        if escalar:

            pasos.append(
                (
                    "escalar",
                    StandardScaler()
                )
            )

        transformaciones.append(
            (
                "numericas",
                Pipeline(pasos),
                numericas
            )
        )

    if categoricas:

        transformaciones.append(
            (
                "categoricas",
                Pipeline([
                    (
                        "imputar",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Desconocido",
                            keep_empty_features=True
                        )
                    ),
                    (
                        "codificar",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]),
                categoricas
            )
        )

    return Pipeline([
        (
            "preparar",
            ColumnTransformer(
                transformaciones
            )
        ),
        (
            "modelo",
            modelo
        )
    ])


# 4. GRÁFICOS.
def preparar_graficas(args):

    import matplotlib

    if not args.graficas:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    return plt


def terminar_figura(
    fig,
    nombre,
    args,
    plt
):

    if args.salida:

        fig.savefig(
            args.salida / f"{nombre}.png",
            dpi=150,
            bbox_inches="tight"
        )

    if not args.graficas:
        plt.close(fig)


def nombres_clases(args):

    if (
        args.objetivo == "Churn"
        and args.clases == ["No", "Yes"]
    ):

        return [
            "Permanece",
            "Abandona"
        ]

    return [
        str(c)
        for c in args.clases
    ]


def graficar_resultados(
    modelos,
    X_train,
    X_test,
    y_test,
    predicciones,
    filas,
    args,
    ejemplo
):

    from sklearn.metrics import ConfusionMatrixDisplay

    plt = preparar_graficas(args)

    colores = [
        "#0072B2",
        "#D55E00",
        "#009E73"
    ]

    estilos = [
        "-",
        "--",
        "-."
    ]

    nombres = list(modelos)

    metricas = {
        f["modelo"]: f
        for f in filas
    }

    fig, (
        a,
        b
    ) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        layout="constrained"
    )

    fig.suptitle(
        "Predicción de abandono de clientes "
        f"· {len(y_test):,} clientes de prueba",
        fontsize=16
    )

    etiquetas = nombres_clases(args)
    clases = np.arange(
        len(etiquetas)
    )

    if len(modelos) == 1:

        n = nombres[0]

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predicciones[n],
            labels=clases,
            display_labels=etiquetas,
            ax=a,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        a.set(
            title="Matriz de confusión del abandono",
            xlabel="Predicción",
            ylabel="Real"
        )

        dibujar_curvas_clasificacion(
            b,
            modelos,
            X_test,
            y_test,
            predicciones,
            args,
            colores,
            estilos
        )

    else:

        dibujar_curvas_clasificacion(
            a,
            modelos,
            X_test,
            y_test,
            predicciones,
            args,
            colores,
            estilos
        )

        posiciones = np.arange(
            len(nombres)
        )

        for j, (
            metrica,
            etiqueta
        ) in enumerate([
            (
                "accuracy",
                "Exactitud"
            ),
            (
                "balanced_accuracy",
                "Exactitud equilibrada"
            ),
            (
                "F1_macro",
                "F1 macro"
            )
        ]):

            barras = b.barh(
                posiciones
                + (j - 1) * .24,
                [
                    metricas[n][metrica]
                    for n in nombres
                ],
                height=.24,
                label=etiqueta,
                color=colores[j]
            )

            b.bar_label(
                barras,
                fmt="%.3f",
                padding=3,
                fontsize=8
            )

        b.set_yticks(
            posiciones,
            [
                n.replace(
                    "_",
                    " "
                )
                for n in nombres
            ],
            fontsize=9
        )

        b.invert_yaxis()

        b.set(
            title="Métricas de clasificación",
            xlabel="Puntuación"
        )

        b.legend(
            loc="upper center",
            bbox_to_anchor=(.5, -.13),
            ncol=3,
            fontsize=8
        )

    nota = (
        "La clase positiva es 'Abandona'. "
        "Las métricas de abandono se calculan sobre "
        "clientes reservados para prueba."
    )

    if NOTA_MODELO:
        nota += "\n" + NOTA_MODELO

    fig.supxlabel(
        nota,
        fontsize=9
    )

    terminar_figura(
        fig,
        "01_resultados",
        args,
        plt
    )


def dibujar_curvas_clasificacion(
    ax,
    modelos,
    X_test,
    y_test,
    predicciones,
    args,
    colores,
    estilos
):

    from sklearn.metrics import (
        roc_curve,
        roc_auc_score
    )

    if len(args.clases) == 2:

        for i, (
            n,
            modelo
        ) in enumerate(
            modelos.items()
        ):

            if hasattr(
                modelo,
                "predict_proba"
            ):

                scores = (
                    modelo.predict_proba(
                        X_test
                    )[
                        :,
                        list(
                            modelo.classes_
                        ).index(1)
                    ]
                )

            elif hasattr(
                modelo,
                "decision_function"
            ):

                scores = (
                    modelo.decision_function(
                        X_test
                    )
                )

            else:

                continue

            fpr, tpr, _ = roc_curve(
                y_test,
                scores,
                pos_label=1
            )

            auc = roc_auc_score(
                y_test,
                scores
            )

            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                color=colores[
                    i % len(colores)
                ],
                linestyle=estilos[
                    i % len(estilos)
                ],
                label=(
                    f"{n.replace('_', ' ')} "
                    f"· AUC={auc:.3f}"
                )
            )

        ax.plot(
            [0, 1],
            [0, 1],
            "--",
            color="gray",
            label="Referencia aleatoria"
        )

        ax.set(
            title="Curva ROC · abandono de clientes",
            xlabel="Proporción de falsos positivos",
            ylabel="Sensibilidad",
            xlim=(0, 1),
            ylim=(0, 1.03)
        )

        ax.legend(
            loc="lower right",
            fontsize=8
        )

    else:

        clases = np.arange(
            len(args.clases)
        )

        ancho = (
            .8 / len(modelos)
        )

        for i, n in enumerate(
            modelos
        ):

            valores = f1_score(
                y_test,
                predicciones[n],
                labels=clases,
                average=None,
                zero_division=0
            )

            ax.bar(
                clases
                + (
                    i
                    - (len(modelos) - 1) / 2
                ) * ancho,
                valores,
                width=ancho,
                color=colores[
                    i % len(colores)
                ],
                label=n.replace(
                    "_",
                    " "
                )
            )

        ax.set_xticks(
            clases,
            nombres_clases(args),
            fontsize=9
        )

        ax.set(
            title="F1 por clase",
            ylabel="F1"
        )

        ax.legend(
            fontsize=8
        )


# 5. EVALUACIÓN.
def particiones_validacion(
    X,
    tarea
):

    return StratifiedKFold(
        3,
        shuffle=True,
        random_state=SEMILLA
    )


def separar_datos(
    X,
    y,
    tarea
):

    return train_test_split(
        X,
        y,
        test_size=.25,
        random_state=SEMILLA,
        stratify=y
    )


def ejecutar(
    archivo,
    args,
    construir,
    ejemplo="general"
):

    X, y, fuente = cargar_datos(
        args,
        ejemplo
    )

    X_train, X_test, y_train, y_test = separar_datos(
        X,
        y,
        args.tarea
    )

    salida = args.salida

    if salida:
        salida.mkdir(
            parents=True,
            exist_ok=True
        )

    print(
        f"Datos: {fuente}\n"
        f"Entrenamiento: {len(X_train)}; "
        f"prueba: {len(X_test)}"
    )

    print(
        "Objetivo:",
        args.objetivo
    )

    print(
        "Entradas:",
        ", ".join(X.columns)
    )

    print(
        "Reparto por clientes; "
        "semilla 42; clasificación estratificada."
    )

    print(
        "Objetivo: identificar clientes "
        "con riesgo de abandono."
    )

    print(
        "Clase 0 = Permanece"
    )

    print(
        "Clase 1 = Abandona"
    )

    if NOTA_MODELO:
        print(NOTA_MODELO)

    modelos = construir(
        X_train,
        args.tarea
    )

    filas = []
    parametros = {}

    predicciones = pd.DataFrame({
        "cliente"
        if getattr(
            args,
            "es_telco",
            False
        )
        else "fila":
            X_test.index,

        "real":
            y_test.to_numpy()
    })

    for nombre, modelo in modelos.items():

        print(
            f"Entrenando {nombre}...",
            flush=True
        )

        modelo.fit(
            X_train,
            y_train
        )

        pred = modelo.predict(
            X_test
        )

        if not np.isfinite(
            pred
        ).all():

            raise ValueError(
                f"{nombre} produjo "
                "predicciones no finitas."
            )

        predicciones[
            nombre
        ] = pred

        metricas = {
            "accuracy":
                accuracy_score(
                    y_test,
                    pred
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_test,
                    pred
                ),

            "F1_macro":
                f1_score(
                    y_test,
                    pred,
                    average="macro",
                    zero_division=0
                )
        }

        # Métricas específicas de abandono.
        if len(args.clases) == 2:

            metricas.update({

                "precision_abandono":
                    precision_score(
                        y_test,
                        pred,
                        pos_label=1,
                        zero_division=0
                    ),

                "recall_abandono":
                    recall_score(
                        y_test,
                        pred,
                        pos_label=1,
                        zero_division=0
                    )
            })

            if hasattr(
                modelo,
                "predict_proba"
            ):

                scores = (
                    modelo.predict_proba(
                        X_test
                    )[
                        :,
                        list(
                            modelo.classes_
                        ).index(1)
                    ]
                )

            else:

                scores = (
                    modelo.decision_function(
                        X_test
                    )
                )

            metricas[
                "ROC_AUC"
            ] = roc_auc_score(
                y_test,
                scores
            )

        reporte = classification_report(
            y_test,
            pred,
            target_names=[
                "Permanece",
                "Abandona"
            ],
            zero_division=0
        )

        print("\nReporte de clasificación:")
        print(reporte)

        if salida:

            (
                salida
                / f"{nombre}_reporte.txt"
            ).write_text(
                reporte,
                encoding="utf-8"
            )

        filas.append({
            "modelo": nombre,
            **metricas
        })

        print(
            nombre,
            {
                k: round(
                    float(v),
                    4
                )
                for k, v in metricas.items()
            }
        )

        if hasattr(
            modelo,
            "best_params_"
        ):

            parametros[
                nombre
            ] = modelo.best_params_

            print(
                "Parámetros elegidos "
                "SOLO en entrenamiento:",
                modelo.best_params_
            )

    if args.graficas or salida:

        graficar_resultados(
            modelos,
            X_train,
            X_test,
            y_test,
            predicciones,
            filas,
            args,
            ejemplo
        )

    if salida:

        pd.DataFrame(
            filas
        ).to_csv(
            salida / "metricas.csv",
            index=False
        )

        predicciones.to_csv(
            salida / "predicciones.csv",
            index=False
        )

        (
            salida / "ejecucion.json"
        ).write_text(
            json.dumps(
                {
                    "fuente": fuente,
                    "dataset": (
                        "Telco"
                        if getattr(
                            args,
                            "es_telco",
                            False
                        )
                        else "otro"
                    ),
                    "tarea":
                        args.tarea,

                    "problematica":
                        "abandono de clientes",

                    "semilla":
                        SEMILLA,

                    "entrenamiento":
                        len(X_train),

                    "prueba":
                        len(X_test),

                    "variables":
                        list(X.columns),

                    "objetivo":
                        args.objetivo,

                    "clase_positiva":
                        "Yes = Abandona",

                    "parametros":
                        parametros,

                    "clases":
                        getattr(
                            args,
                            "clases",
                            None
                        ),

                    "reparto":
                        "estratificado_por_cliente"
                },
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            "Resultados guardados:",
            salida.resolve()
        )

    if args.graficas:

        import matplotlib.pyplot as plt

        print(
            "Cierra las ventanas de gráficos "
            "para terminar.",
            flush=True
        )

        plt.show()
        plt.close("all")


# 6. MODELO DE REGRESIÓN LOGÍSTICA.
from sklearn.linear_model import LogisticRegression


def construir(
    X,
    tarea
):

    """
    Predice el abandono de clientes
    mediante regresión logística.

    C controla la regularización:
    menor C = mayor regularización.
    """

    modelo = preparar(
        X,
        LogisticRegression(
            max_iter=5000
        ),
        escalar=True
    )

    busqueda = GridSearchCV(
        modelo,
        {
            "modelo__C": [
                0.01,
                0.1,
                1,
                10
            ]
        },
        cv=particiones_validacion(
            X,
            tarea
        ),
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise"
    )

    return {
        "logistica": busqueda
    }


# 7. INICIO.
if __name__ == "__main__":

    ejecutar(
        __file__,
        argumentos(
            __doc__,
            ["clasificacion"]
        ),
        construir
    )