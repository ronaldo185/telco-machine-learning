"""Comparación Bagging (Random Forest) vs Boosting, sin aplicación web.

Archivo independiente. Ejecución: python modelo.py
Con tus datos: python modelo.py --datos "datos.xlsx" --objetivo "precio" --tarea regresion
También acepta .csv. Usa --hoja Nombre para otra hoja y --separador ";" si hace falta.
Las rutas relativas del dataset se buscan junto a este archivo.
Por defecto usa telco.csv de esta carpeta (7.043 filas).
El objetivo se elige según --tarea. --demo permite volver al ejemplo pequeño.
No se crean archivos por defecto; --salida permite guardarlos.
Una figura por ejecución; polinomial y árboles: dos. Se muestran al terminar.
--sin-graficas evita abrir ventanas; no se cambia ningún ajuste para forzar curvas.
Opcional: --salida CARPETA guarda los resultados y las figuras PNG.
Instalación: python -m pip install numpy pandas scipy scikit-learn matplotlib openpyxl
Selecciona variables disponibles antes de predecir; excluye identificadores y
columnas derivadas del objetivo. Telco: clientes reservados, sin separación temporal.
Clasificación: Churn. Regresión: MonthlyCharges, sin TotalCharges ni Churn como entradas.
"""

# 1. CONFIGURACIÓN: dataset, objetivo y opciones de ejecución.
RUTA_DATASET = "telco.csv"  # Se busca junto a modelo.py.
COLUMNA_OBJETIVO = None   # Telco: MonthlyCharges o Churn según la tarea.
TAREA = "clasificacion"  # Consulta las tareas admitidas al final o con --help.
MOSTRAR_GRAFICAS = True  # Cierra las ventanas para terminar el programa.
HOJA_EXCEL = 0            # Primera hoja, o su nombre entre comillas.
COLUMNAS_ENTRADA = []     # Telco: selección automática según la tarea.

NOTA_MODELO = 'Random Forest: bagging; AdaBoost y Gradient Boosting: boosting.'

# Bibliotecas para datos, preparación y evaluación.
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_wine
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

SEMILLA = 42


# 2. DATOS: lectura, opciones y validación.
def leer_tabla(ruta, separador=None, hoja=0):
    """Lee CSV o Excel y resuelve las rutas desde la carpeta del programa."""
    ruta = Path(ruta).expanduser()
    # Las rutas relativas se buscan junto a este modelo.py.
    if not ruta.is_absolute():
        ruta = Path(__file__).resolve().parent / ruta
    if not ruta.is_file():
        raise ValueError(f"No se encontró el dataset: {ruta}")
    if ruta.suffix.lower() == ".csv":
        tabla = pd.read_csv(ruta, sep=separador, engine="python", encoding="utf-8-sig")
    elif ruta.suffix.lower() == ".xlsx":
        tabla = pd.read_excel(ruta, sheet_name=hoja, engine="openpyxl")
    else:
        raise ValueError("El dataset debe ser un archivo .csv o .xlsx.")
    if tabla.empty:
        raise ValueError("El dataset está vacío.")
    tabla.columns = tabla.columns.map(str)
    return tabla, str(ruta.resolve())


def argumentos(titulo, tareas):
    """Recoge las opciones de ejecución; los valores iniciales están arriba."""
    parser = argparse.ArgumentParser(description=titulo)
    parser.add_argument("--tarea", choices=tareas, default=TAREA)
    parser.add_argument("--datos", "--csv", "--xlsx", dest="datos", default=RUTA_DATASET,
                        help="Ruta al dataset .csv o .xlsx (relativa a modelo.py o absoluta)")
    parser.add_argument("--objetivo", default=COLUMNA_OBJETIVO, help="Columna que se predice")
    parser.add_argument("--hoja", default=HOJA_EXCEL, help="Nombre o índice de hoja Excel; primera: 0")
    parser.add_argument("--separador", default=None, help="Separador CSV; por defecto se detecta")
    parser.add_argument("--columnas", nargs="+", default=COLUMNAS_ENTRADA or None,
                        help="Variables que se usarán; omitir para usar todas salvo el objetivo")
    parser.add_argument("--salida", type=Path, help="Opcional: guardar resultados en esta carpeta")
    visual = parser.add_mutually_exclusive_group()
    visual.add_argument("--graficas", dest="graficas", action="store_true", help="Mostrar gráficos (predeterminado)")
    visual.add_argument("--sin-graficas", dest="graficas", action="store_false", help="No abrir ventanas; con --salida guarda PNG")
    parser.set_defaults(graficas=MOSTRAR_GRAFICAS)
    parser.add_argument("--demo", action="store_true", help="Ejecutar el ejemplo pequeño incluido")
    args = parser.parse_args()
    if args.demo:
        args.datos = None
        args.objetivo = None
        args.columnas = None
    if args.tarea not in tareas:
        parser.error(f"TAREA debe ser una de: {tareas}")
    args.hoja = int(args.hoja) if str(args.hoja).isdigit() else args.hoja
    return args


def cargar_datos(args, ejemplo="general"):
    """Separa entradas y objetivo, valida valores y codifica las clases."""
    if args.datos:
        tabla, fuente = leer_tabla(args.datos, args.separador, args.hoja)
        # Telco: conservar filas, convertir cargos y excluir identificadores/fugas.
        args.es_telco = {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"}.issubset(tabla.columns)
        if args.es_telco:
            tabla = tabla.copy()
            if tabla.customerID.isna().any() or tabla.customerID.duplicated().any():
                raise ValueError("Cada cliente debe tener un identificador único y no vacío.")
            for col in tabla.select_dtypes(exclude="number"):
                tabla[col] = tabla[col].astype("string").str.strip().replace("", np.nan)
            for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                tabla[col] = pd.to_numeric(tabla[col], errors="raise").astype(float)
                if np.isinf(tabla[col]).any() or tabla[col].dropna().lt(0).any():
                    raise ValueError(f"Valores inválidos en {col}.")
            if not tabla.Churn.isin(["Yes", "No"]).all():
                raise ValueError("Churn debe contener Yes o No, sin vacíos.")
            if "SeniorCitizen" in tabla:
                if not tabla.SeniorCitizen.isin([0, 1]).all():
                    raise ValueError("SeniorCitizen debe ser 0 o 1.")
                tabla["SeniorCitizen"] = tabla.SeniorCitizen.map({0: "No", 1: "Yes"})
            esperado = "MonthlyCharges" if args.tarea == "regresion" else "Churn"
            if args.objetivo is not None and args.objetivo != esperado:
                raise ValueError(f"Para Telco y {args.tarea}, utiliza el objetivo {esperado}.")
            args.objetivo = esperado
            excluir = {"customerID", "Churn"}
            if args.tarea == "regresion":
                excluir |= {"MonthlyCharges", "TotalCharges"}
            permitidas = ["tenure"] + [c for c in tabla if c not in excluir | {"tenure"}]
            args.columnas = permitidas if args.columnas is None else args.columnas
            if not set(args.columnas).issubset(permitidas):
                raise ValueError("Excluir customerID, el objetivo y variables derivadas del objetivo.")
            tabla = tabla.set_index("customerID")
            print("Telco: TotalCharges faltantes se imputan con la mediana SOLO del entrenamiento.")
        elif args.objetivo is None:
            raise ValueError("Para otro dataset indica --objetivo o configura COLUMNA_OBJETIVO.")
        if args.objetivo not in tabla:
            raise ValueError(f"No existe la columna objetivo '{args.objetivo}'. Columnas: {list(tabla.columns)}")
        if tabla[args.objetivo].isna().any():
            raise ValueError("La columna objetivo contiene valores vacíos.")
        X, y = tabla.drop(columns=args.objetivo), tabla[args.objetivo]
    elif ejemplo == "polinomial":
        rng = np.random.default_rng(SEMILLA)
        x = rng.uniform(-3, 3, 240)
        X = pd.DataFrame({"x": x})
        y = pd.Series(0.5*x**2 - 3*x + 5 + rng.normal(0, 1, len(x)))
        fuente = "DEMOSTRACIÓN: parábola sintética, semilla 42"
    else:
        datos = load_wine(as_frame=True) if args.tarea == "clasificacion" else load_diabetes(as_frame=True)
        X, y = datos.data, datos.target
        fuente = "DEMOSTRACIÓN: Wine de scikit-learn" if args.tarea == "clasificacion" else "DEMOSTRACIÓN: Diabetes de scikit-learn"
    if args.columnas:
        faltantes = set(args.columnas) - set(X.columns)
        if faltantes:
            raise ValueError(f"Variables inexistentes o iguales al objetivo: {sorted(faltantes)}")
        if len(args.columnas) != len(set(args.columnas)):
            raise ValueError("No repitas nombres en las columnas de entrada.")
        X = X[args.columnas].copy()
    if X.shape[1] == 0 or len(X) < 20:
        raise ValueError("Se necesitan al menos 20 filas y una variable predictora.")
    if np.isinf(X.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("Hay valores infinitos en las variables predictoras.")
    # Las categorías se representan como texto conservando los nulos.
    for columna in X.select_dtypes(exclude="number").columns:
        X[columna] = X[columna].map(lambda v: str(v) if pd.notna(v) else np.nan)
    # Regresión requiere números; clasificación codifica nombres de clases.
    if args.tarea == "regresion":
        y = pd.to_numeric(y, errors="raise")
        if not np.isfinite(y).all() or y.nunique() < 2:
            raise ValueError("El objetivo de regresión debe contener números finitos y variar.")
    else:
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)), index=X.index, name=y.name)
        if y.nunique() < 2 or y.value_counts().min() < 8:
            raise ValueError("Se requieren al menos dos clases y ocho filas por clase.")
        args.clases = [str(c) for c in encoder.classes_]
        clases = dict(enumerate(encoder.classes_))
        print("Clases:", clases)
        fuente += "; clases=" + str(clases)
    return X, y, fuente


# 3. PREPARACIÓN: cada transformación aprende solo del entrenamiento.
def preparar(X, modelo, escalar=False):
    """Une imputación, codificación y modelo en un único ajuste de entrenamiento."""
    # Estos pasos aprenden SOLO del entrenamiento, incluso dentro de la validación.
    numericas = list(X.select_dtypes(include="number").columns)
    categoricas = [c for c in X.columns if c not in numericas]
    pasos = [("imputar", SimpleImputer(strategy="median", keep_empty_features=True))]
    if escalar:
        pasos.append(("escalar", StandardScaler()))
    transformaciones = []
    if numericas:
        transformaciones.append(("numericas", Pipeline(pasos), numericas))
    if categoricas:
        transformaciones.append(("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido", keep_empty_features=True)),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categoricas))
    return Pipeline([("preparar", ColumnTransformer(transformaciones)), ("modelo", modelo)])


# 4. GRÁFICOS: resumen del ajuste y evaluación sobre datos reservados.
def preparar_graficas(args):
    """Configura las figuras para mostrarlas o guardarlas sin ventanas."""
    import matplotlib
    if not args.graficas:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def terminar_figura(fig, nombre, args, plt):
    """Guarda solo si se pidió --salida; conserva las ventanas hasta el final."""
    if args.salida:
        fig.savefig(args.salida / f"{nombre}.png", dpi=150, bbox_inches="tight")
    if not args.graficas:
        plt.close(fig)


def nombres_clases(args):
    """Etiqueta positiva de Telco: el cliente abandonó el servicio."""
    import textwrap
    if args.objetivo == "Churn" and args.clases == ["No", "Yes"]:
        return ["Permanece", "Abandona"]
    return [textwrap.fill(str(c), 18) for c in args.clases]


def crear_perfil(X_train, variable=None):
    """Varía tenure sobre un cliente de entrenamiento; no representa un efecto causal."""
    numericas = list(X_train.select_dtypes(include="number").columns)
    candidatas = [c for c in numericas if X_train[c].nunique() > 1]
    if not candidatas:
        return None, None
    variable = variable or ("tenure" if "tenure" in candidatas else candidatas[0])
    # Una fila real mantiene coherentes las categorías de servicios.
    centro = X_train[variable].median()
    base = X_train.loc[[(X_train[variable]-centro).abs().idxmin()]].reset_index(drop=True)
    minimo, maximo = X_train[variable].quantile([.01, .99])
    perfil = pd.concat([base] * 160, ignore_index=True)
    perfil[variable] = np.linspace(minimo, maximo, len(perfil))
    return variable, perfil


def dibujar_validacion_knn(ax, buscador, tarea, colores):
    """Muestra cómo se eligió K usando únicamente particiones del entrenamiento."""
    cv = buscador.cv_results_
    k = [p["modelo__n_neighbors"] for p in cv["params"]]
    # sklearn maximiza puntuaciones; neg_mean_squared_error se convierte a error positivo.
    medias = np.asarray(cv["mean_test_score"])
    if tarea == "regresion":
        medias = -medias
    ax.errorbar(k, medias, yerr=cv["std_test_score"], fmt="o-", capsize=4, color=colores[0])
    mejor = buscador.best_index_
    ax.scatter(k[mejor], medias[mejor], s=80, color=colores[1], label=f"K elegido = {k[mejor]}")
    etiqueta = "ECM: menor es mejor" if tarea == "regresion" else "F1 macro: mayor es mejor"
    ax.set(title="Selección de K: solo entrenamiento", xlabel="Número de vecinos K", ylabel=etiqueta)
    ax.legend(fontsize=9)


def graficar_resultados(modelos, X_train, X_test, y_test, predicciones, filas, args, ejemplo):
    """Una figura con dos paneles; polinomial y árboles usan dos figuras como máximo."""
    from sklearn.metrics import ConfusionMatrixDisplay
    plt = preparar_graficas(args)
    colores = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6F5AA8"]
    estilos = ["-", "--", "-.", ":", (0, (5, 2))]
    nombres = list(modelos)
    metricas = {f["modelo"]: f for f in filas}
    unidad = " (u. monetarias/mes)" if args.objetivo == "MonthlyCharges" else ""
    tarea_visible = "regresión" if args.tarea == "regresion" else "clasificación"
    if args.tarea == "regresion":
        variable, perfil = crear_perfil(X_train)
        curvas = {n: m.predict(perfil) for n, m in modelos.items()} if perfil is not None else {}

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    fig.suptitle(f"Resultados de {tarea_visible} · {len(y_test):,} filas de prueba", fontsize=16)
    if args.tarea == "regresion":
        if "knn" in modelos and hasattr(modelos["knn"], "cv_results_"):
            dibujar_validacion_knn(a, modelos["knn"], args.tarea, colores)
            nota = "K se elige dentro del entrenamiento; barras: ±1 desviación entre particiones. Derecha: prueba reservada."
        elif any(n in modelos for n in ["gradient_boosting", "adaboost", "random_forest"]):
            # Para conjuntos, comparar todos los errores de prueba es más informativo que un perfil aislado.
            posiciones = np.arange(len(nombres))
            for j, clave in enumerate(["MAE", "RMSE"]):
                barras = a.barh(posiciones + (j - .5) * .34, [metricas[n][clave] for n in nombres],
                                height=.34, label=clave, color=colores[j])
                a.bar_label(barras, fmt="%.3f", padding=3, fontsize=8)
            a.set_yticks(posiciones, [n.replace('_', ' ') for n in nombres], fontsize=9)
            a.invert_yaxis()
            a.margins(x=.2)
            a.set(title="Errores de TODAS las variantes", xlabel="Error: menor es mejor" + unidad)
            nota = "MAE y RMSE se calculan sobre las mismas filas de prueba, en las unidades del objetivo."
        elif perfil is not None:
            for i, n in enumerate(nombres):
                a.plot(perfil[variable], curvas[n], color=colores[i % len(colores)], linestyle=estilos[i % len(estilos)],
                       linewidth=2, label=f"{n.replace('_', ' ')} · RMSE={metricas[n]['RMSE']:.3f}")
            a.set(title="Comparación de perfiles del modelo", xlabel=variable, ylabel="Predicción" + unidad)
            nota = "Izquierda: varía una entrada; las demás quedan fijas en un cliente de entrenamiento. No indica causalidad."
        else:
            for i, n in enumerate(nombres):
                a.hist(np.asarray(y_test) - predicciones[n].to_numpy(), bins=25, alpha=.4, label=n.replace('_', ' '), color=colores[i % len(colores)])
            a.axvline(0, color="gray", linestyle="--")
            a.set(title="Errores por variante", xlabel="Real − predicción" + unidad, ylabel="Filas de prueba")
            nota = "Errores cercanos a cero y puntos cercanos a la diagonal indican mejores predicciones."
        a.legend(fontsize=8)
        # Referencia fija, elegida por tema; nunca se escoge usando el resultado de prueba.
        referencias = ["lineal_multiple", "svm_rbf", "gradient_boosting"]
        principal = next((n for n in referencias if n in modelos), nombres[0])
        pred = predicciones[principal].to_numpy()
        b.scatter(y_test, pred, s=10, alpha=.25, color=colores[0], rasterized=True)
        limites = [min(np.min(y_test), np.min(pred)), max(np.max(y_test), np.max(pred))]
        b.plot(limites, limites, "--", color="#D55E00", label="Predicción perfecta")
        m = metricas[principal]
        b.set(title=f"{principal.replace('_', ' ')}: real frente a predicción\nMAE={m['MAE']:.3f} · RMSE={m['RMSE']:.3f} · R²={m['R2']:.3f}",
              xlabel="Valor real" + unidad, ylabel="Predicción" + unidad)
        b.set_xlim(limites)
        b.set_ylim(limites)
        b.set_aspect("equal", adjustable="box")
        b.legend(fontsize=8)
        if len(modelos) > 1:
            nota += "\nDerecha: una variante de referencia; las métricas de TODAS las variantes se muestran en la consola."
    else:
        etiquetas = nombres_clases(args)
        clases = np.arange(len(etiquetas))
        if len(modelos) == 1:
            n = nombres[0]
            ConfusionMatrixDisplay.from_predictions(y_test, predicciones[n], labels=clases, display_labels=etiquetas,
                ax=a, cmap="Blues", colorbar=False, values_format="d")
            a.set(title="Matriz de confusión: aciertos en diagonal", xlabel="Clase predicha", ylabel="Clase real")
            a.tick_params(labelsize=9)
            buscador = modelos[n]
            if n == "knn" and hasattr(buscador, "cv_results_"):
                dibujar_validacion_knn(b, buscador, args.tarea, colores)
                nota = "La matriz usa la prueba reservada; K usa validación en entrenamiento. Barras: ±1 desviación entre particiones."
            else:
                dibujar_curvas_clasificacion(b, modelos, X_test, y_test, predicciones, args, colores, estilos)
                nota = "Evaluación con filas de prueba reservadas; las métricas completas aparecen en la consola."
        else:
            dibujar_curvas_clasificacion(a, modelos, X_test, y_test, predicciones, args, colores, estilos)
            posiciones = np.arange(len(nombres))
            for j, (metrica, etiqueta) in enumerate([("accuracy", "Exactitud"), ("balanced_accuracy", "Exactitud equilibrada"), ("F1_macro", "F1 macro")]):
                b.barh(posiciones + (j - 1) * .24, [metricas[n][metrica] for n in nombres], height=.24, label=etiqueta, color=colores[j])
            b.set_yticks(posiciones, [n.replace('_', ' ') for n in nombres], fontsize=9)
            b.invert_yaxis()
            b.set(title="Comparación sobre las mismas filas de prueba", xlabel="Puntuación (mayor es mejor)", xlim=(0, 1.03))
            b.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, fontsize=8)
            nota = "Las curvas y las puntuaciones usan exclusivamente las filas reservadas para prueba."
        if len(etiquetas) == 2 and not (len(modelos) == 1 and nombres[0] == "knn"):
            nota += f"\nEn las curvas ROC, la clase positiva es: {etiquetas[1]}."
    if NOTA_MODELO:
        nota += "\n" + NOTA_MODELO
    fig.supxlabel(nota, fontsize=9)
    terminar_figura(fig, "01_resultados", args, plt)

    # Solo el árbol individual requiere una segunda figura: total de tres gráficos.
    for n, modelo in modelos.items():
        elegido = getattr(modelo, "best_estimator_", modelo)
        if isinstance(elegido, Pipeline) and hasattr(elegido.steps[-1][1], "tree_"):
            from sklearn.tree import plot_tree
            fig, ax = plt.subplots(figsize=(16, 8), layout="constrained")
            entradas = [str(c).split("__", 1)[-1] for c in elegido[:-1].get_feature_names_out()]
            plot_tree(elegido.steps[-1][1], max_depth=2, feature_names=entradas,
                class_names=nombres_clases(args) if args.tarea == "clasificacion" else None,
                filled=True, rounded=True, precision=2, fontsize=8, ax=ax)
            leyenda = "samples = filas; value = conteos por clase; class = clase mayoritaria"
            if args.tarea == "regresion":
                leyenda = "samples = filas; value = promedio del objetivo" + unidad
            ax.set_title("Árbol · primeras tres capas; (…) indica continuación\n" + leyenda, fontsize=12)
            terminar_figura(fig, "02_reglas_arbol", args, plt)
            break


def dibujar_curvas_clasificacion(ax, modelos, X_test, y_test, predicciones, args, colores, estilos):
    """Evalúa la separación entre clases con ROC binaria o F1 multiclase."""
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score
    if len(args.clases) == 2:
        for i, (n, modelo) in enumerate(modelos.items()):
            # ROC recorre umbrales; decision_function entrega márgenes, no probabilidades.
            if hasattr(modelo, "predict_proba"):
                scores = modelo.predict_proba(X_test)[:, list(modelo.classes_).index(1)]
            elif hasattr(modelo, "decision_function"):
                scores = modelo.decision_function(X_test)
            else:
                continue
            fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
            ax.plot(fpr, tpr, linewidth=2, color=colores[i % len(colores)], linestyle=estilos[i % len(estilos)],
                    label=f"{n.replace('_', ' ')} · AUC={roc_auc_score(y_test, scores):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Referencia aleatoria")
        ax.set(title="Curvas ROC: comparación de clasificación", xlabel="Proporción de falsos positivos", ylabel="Sensibilidad", xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(loc="lower right", fontsize=8)
    else:
        clases = np.arange(len(args.clases))
        ancho = .8 / len(modelos)
        for i, n in enumerate(modelos):
            valores = f1_score(y_test, predicciones[n], labels=clases, average=None, zero_division=0)
            ax.bar(clases + (i - (len(modelos) - 1) / 2) * ancho, valores, width=ancho, color=colores[i % len(colores)], label=n.replace('_', ' '))
        ax.set_xticks(clases, nombres_clases(args), fontsize=9)
        ax.set(title="F1 por clase", ylabel="F1 (mayor es mejor)", ylim=(0, 1.2))
        ax.legend(fontsize=8)


# 5. EVALUACIÓN: mismo reparto para comparar las variantes.
def particiones_validacion(X, tarea):
    """Tres particiones internas; en clasificación conservan la proporción de clases."""
    return (StratifiedKFold if tarea == "clasificacion" else KFold)(
        3, shuffle=True, random_state=SEMILLA)


def separar_datos(X, y, tarea):
    """Reserva 25 % de clientes. No hay fechas para una validación temporal."""
    return train_test_split(X, y, test_size=.25, random_state=SEMILLA,
                            stratify=y if tarea == "clasificacion" else None)



def ejecutar(archivo, args, construir, ejemplo="general"):
    """Entrena sin consultar la prueba y compara todas las variantes en los mismos clientes."""
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    X, y, fuente = cargar_datos(args, ejemplo)
    X_train, X_test, y_train, y_test = separar_datos(X, y, args.tarea)
    salida = args.salida
    if salida:
        salida.mkdir(parents=True, exist_ok=True)
    print(f"Datos: {fuente}\nEntrenamiento: {len(X_train)}; prueba: {len(X_test)}")
    print("Objetivo:", args.objetivo or "objetivo del ejemplo")
    print("Entradas:", ", ".join(X.columns))
    print("Reparto por clientes; semilla 42; clasificación estratificada. No es un pronóstico temporal.")
    if args.objetivo == "MonthlyCharges":
        print("MAE y RMSE: unidades monetarias del cargo mensual. No se usan TotalCharges ni Churn como entradas.")
    if NOTA_MODELO:
        print(NOTA_MODELO)
    modelos = construir(X_train, args.tarea)
    filas, parametros = [], {}
    predicciones = pd.DataFrame({"cliente" if getattr(args, "es_telco", False) else "fila": X_test.index,
                                "real": y_test.to_numpy()})
    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...", flush=True)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        if not np.isfinite(pred).all():
            raise ValueError(f"{nombre} produjo predicciones no finitas.")
        predicciones[nombre] = pred
        if args.tarea == "regresion":
            metricas = {"MAE": mean_absolute_error(y_test, pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)}
        else:
            metricas = {"accuracy": accuracy_score(y_test, pred),
                        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                        "F1_macro": f1_score(y_test, pred, average="macro", zero_division=0)}
            if len(args.clases) == 2:
                metricas.update(precision_abandono=precision_score(y_test, pred, zero_division=0),
                                recall_abandono=recall_score(y_test, pred, zero_division=0))
                scores = (modelo.predict_proba(X_test)[:, list(modelo.classes_).index(1)]
                          if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test))
                metricas["ROC_AUC"] = roc_auc_score(y_test, scores)
            reporte = classification_report(y_test, pred, target_names=args.clases, zero_division=0)
            print(reporte)
            if salida:
                (salida / f"{nombre}_reporte.txt").write_text(reporte, encoding="utf-8")
        filas.append({"modelo": nombre, **metricas})
        print(nombre, {k: round(float(v), 4) for k, v in metricas.items()})
        if hasattr(modelo, "best_params_"):
            parametros[nombre] = modelo.best_params_
            print("Parámetros elegidos SOLO en entrenamiento:", modelo.best_params_)
    if args.graficas or salida:
        graficar_resultados(modelos, X_train, X_test, y_test, predicciones, filas, args, ejemplo)
    if salida:
        pd.DataFrame(filas).to_csv(salida / "metricas.csv", index=False)
        predicciones.to_csv(salida / "predicciones.csv", index=False)
        (salida / "ejecucion.json").write_text(json.dumps({
            "fuente": fuente, "dataset": "Telco" if getattr(args, "es_telco", False) else "otro",
            "tarea": args.tarea, "semilla": SEMILLA, "entrenamiento": len(X_train), "prueba": len(X_test),
            "variables": list(X.columns), "objetivo": args.objetivo, "parametros": parametros,
            "clases": getattr(args, "clases", None),
            "reparto": "estratificado_por_cliente" if args.tarea == "clasificacion" else "aleatorio_por_cliente",
            "referencia": {"accuracy_siempre_mayoritaria": float((y_test == y_train.mode().iloc[0]).mean())}
                if args.tarea == "clasificacion" else {"MAE_media_entrenamiento": float(np.abs(y_test-y_train.mean()).mean())},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Resultados guardados:", salida.resolve())
    if args.graficas:
        import matplotlib.pyplot as plt
        print("Cierra las ventanas de gráficos para terminar.", flush=True)
        plt.show()
        plt.close("all")


# 6. MODELOS DE ESTE TEMA
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor)


def construir(X, tarea):
    """Compara árboles en paralelo (bosque) y correcciones sucesivas (boosting)."""
    # Random Forest usa remuestreo y variables aleatorias; los otros dos ajustan etapas sucesivas.
    tipos = (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier) if tarea == "clasificacion" else (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
    return {nombre: preparar(X, tipo(n_estimators=100, random_state=SEMILLA))
            for nombre, tipo in zip(["bagging_random_forest", "gradient_boosting", "adaboost"], tipos)}


# 7. INICIO: ejecutar solo al abrir este archivo como programa.
if __name__ == "__main__":
    ejecutar(__file__, argumentos(__doc__, ["clasificacion", "regresion"]), construir)
