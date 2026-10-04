"""A/B testing y comparación observacional de dos proporciones.

Ejecutar: python modelo.py
Telco por defecto: comparación descriptiva de abandono según tipo de contrato.
Los contratos no fueron asignados al azar: no se estima un efecto causal.
Experimento real: python modelo.py --modo experimento --datos "experimento.xlsx"
CSV y XLSX: deben incluir grupo (A/B) y conversion (0/1), una fila por unidad.
Otros nombres: --grupo variante --conversion compra
Excel: --hoja Nombre. CSV: separador detectado o --separador ";".
Rutas relativas del dataset: junto a este modelo.py.
Simulación opcional: python modelo.py --modo demo. No genera archivos por defecto.
Los gráficos se muestran al terminar. --sin-graficas evita abrir ventanas.
Opcional: --salida CARPETA guarda el resultado y las figuras PNG.
Instalar: python -m pip install numpy pandas scipy openpyxl matplotlib
Supuestos: asignación aleatoria, unidades independientes y análisis al cierre.
"""

# 1. CONFIGURACIÓN: fuente de datos y significado de los grupos.
RUTA_DATASET = "telco.csv"  # Copia local, junto a este programa.
MODO = "observacional"   # Telco no contiene un experimento A/B.
COLUMNA_GRUPO = "grupo"   # Valores A y B.
COLUMNA_CONVERSION = "conversion"  # Valores 0 y 1.
HOJA_EXCEL = 0
MOSTRAR_GRAFICAS = True  # Cierra las ventanas para terminar el programa.

# Bibliotecas para tablas, pruebas estadísticas y resultados.
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

# 2. DATOS: aceptar CSV y Excel desde la carpeta del programa.
def leer_tabla(ruta, separador=None, hoja=0):
    """Lee la tabla sin modificar su contenido."""
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


# 3. ANÁLISIS: proporciones, intervalos y contraste bilateral.
def intervalo_wilson(exitos, total, confianza=0.95):
    """Estima el intervalo de una proporción, incluso cerca de cero o uno."""
    z = norm.ppf((1 + confianza) / 2)
    p = exitos / total
    divisor = 1 + z*z/total
    centro = (p + z*z/(2*total)) / divisor
    radio = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / divisor
    return centro-radio, centro+radio


def analizar(tabla, alpha=0.05):
    """Compara dos grupos con resultado binario; no ajusta una regresión."""
    # El cálculo exige dos grupos no vacíos y resultados 0/1.
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1.")
    if not {"grupo", "conversion"}.issubset(tabla.columns):
        raise ValueError("La tabla debe incluir grupo y conversion.")
    if tabla[["grupo", "conversion"]].isna().any().any():
        raise ValueError("No se admiten grupos ni conversiones vacíos.")
    if set(tabla.grupo) != {"A", "B"}:
        raise ValueError("Deben existir exactamente los grupos A y B.")
    if not tabla.conversion.isin([0, 1]).all():
        raise ValueError("conversion debe contener únicamente 0 o 1.")
    resumen = tabla.groupby("grupo").conversion.agg(["sum", "count"])
    a, na = map(int, resumen.loc["A"])
    b, nb = map(int, resumen.loc["B"])
    pa, pb = a/na, b/nb
    # Fisher bilateral funciona también cuando los conteos son pequeños.
    pvalor = float(fisher_exact([[a, na-a], [b, nb-b]], alternative="two-sided").pvalue)
    la, ua = intervalo_wilson(a, na, 1-alpha)
    lb, ub = intervalo_wilson(b, nb, 1-alpha)
    diferencia = pb-pa
    # Intervalo de Newcombe (Wilson sin corrección de continuidad) para B-A.
    inferior = diferencia - math.sqrt((pb-lb)**2 + (ua-pa)**2)
    superior = diferencia + math.sqrt((ub-pb)**2 + (pa-la)**2)
    return {
        "n_A": na, "n_B": nb, "conversiones_A": a, "conversiones_B": b,
        "tasa_A": pa, "tasa_B": pb, "diferencia_B_menos_A": diferencia,
        "cambio_relativo": (pb/pa-1) if pa else None,
        "intervalo_diferencia": [inferior, superior],
        "confianza": 1-alpha, "p_valor_fisher_bilateral": pvalor,
        "alpha": alpha, "diferencia_significativa": bool(pvalor < alpha),
    }


# 4. GRÁFICOS: tasas por grupo y diferencia B − A, con incertidumbre.
def comparar_telco(tabla, fuente, args):
    """Describe abandono por contrato. Una diferencia observada no demuestra causalidad."""
    necesarias = ["customerID", "Contract", "Churn"]
    if not set(necesarias).issubset(tabla.columns) or tabla[necesarias].isna().any().any():
        raise ValueError("Se necesitan customerID, Contract y Churn sin vacíos.")
    if tabla.customerID.duplicated().any() or not tabla.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Identificadores duplicados o Churn inválido.")
    if not tabla.Contract.isin(["Month-to-month", "One year", "Two year"]).all():
        raise ValueError("Tipos de contrato no reconocidos.")
    tabla = tabla.copy()
    tabla["grupo"] = np.where(tabla.Contract.eq("Month-to-month"), "A", "B")
    tabla["abandono"] = tabla.Churn.eq("Yes").astype(int)
    resumen = tabla.groupby("grupo").agg(filas=("abandono", "size"), abandonos=("abandono", "sum"),
                                        proporcion_abandono=("abandono", "mean"))
    if set(resumen.index) != {"A", "B"}:
        raise ValueError("Se necesitan contratos mensuales y de uno/dos años.")
    etiquetas = {"A": "Mes a mes", "B": "Uno o dos años"}
    resultado = {"fuente": fuente, "modo": "observacional", "filas": len(tabla),
        "grupos": resumen.to_dict(orient="index"), "etiquetas": etiquetas,
        "diferencia_proporcion_B_menos_A": float(resumen.loc["B", "proporcion_abandono"]-resumen.loc["A", "proporcion_abandono"]),
        "interpretacion": "Asociación descriptiva, sin asignación aleatoria. No demuestra causalidad ni es un experimento A/B."}
    print("COMPARACIÓN OBSERVACIONAL: contratos mensuales frente a contratos de uno/dos años.")
    print(resumen.to_string())
    print(resultado["interpretacion"])
    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)
        (args.salida/"resultado_ab.json").write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.graficas or args.salida:
        import matplotlib
        if not args.graficas: matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        barras = ax.bar([etiquetas[g] for g in resumen.index], resumen.proporcion_abandono*100,
                        color=["#167d9a", "#e2843c"])
        ax.bar_label(barras, labels=[f"{100*r.proporcion_abandono:.1f}%\n{int(r.abandonos)}/{int(r.filas)} clientes"
                                   for _,r in resumen.iterrows()], padding=5)
        ax.set(title="Telco: abandono observado según contrato", ylabel="Clientes que abandonaron (%)", ylim=(0,100))
        fig.supxlabel("Grupos observados, no asignados al azar. La diferencia no es un efecto causal del contrato.", fontsize=10)
        if args.salida: fig.savefig(args.salida/"comparacion_descriptiva.png", dpi=150)
        if args.graficas: plt.show()
        plt.close(fig)


def graficar(resultado, args):
    """Resume los dos grupos sin presentar asociación como causalidad."""
    import matplotlib
    if not args.graficas:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = resultado
    fig, (a, c) = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    titulo = "Comparación de proporciones A/B"
    if args.modo == "demo":
        titulo += " · datos simulados"
    fig.suptitle(titulo, fontsize=16, fontweight="bold")
    grupos = ["A", "B"]
    tasas = np.array([r[f"tasa_{g}"] for g in grupos])
    intervalos = np.array([intervalo_wilson(r[f"conversiones_{g}"], r[f"n_{g}"], r["confianza"]) for g in grupos])
    error = np.maximum(0, np.array([tasas - intervalos[:, 0], intervalos[:, 1] - tasas])) * 100
    etiquetas = ["Grupo A", "Grupo B"]
    barras = a.bar(etiquetas, tasas * 100, yerr=error, capsize=6, color=["#167d9a", "#e2843c"])
    a.bar_label(barras, fmt="%.1f%%", padding=10)
    a.set(title=f"Proporciones e intervalos de Wilson ({r['confianza']:.0%})",
          ylabel="Resultados positivos (%)",
          ylim=(0, 115))
    dif = r["diferencia_B_menos_A"] * 100
    # La línea en cero indica ausencia de diferencia; la barra muestra el intervalo.
    bajo, alto = np.array(r["intervalo_diferencia"]) * 100
    c.errorbar([dif], [0], xerr=[[dif - bajo], [alto - dif]], fmt="o", capsize=8, color="#167d9a")
    c.axvline(0, color="gray", linestyle="--", label="Sin diferencia")
    margen = max(2, (alto - bajo) * .5)
    c.set(xlim=(min(0, bajo) - margen, max(0, alto) + margen), yticks=[],
          xlabel="Diferencia B − A (puntos porcentuales)",
          title=f"Diferencia: {dif:.2f} puntos\nIntervalo de Newcombe: [{bajo:.2f}, {alto:.2f}]")
    c.legend()
    nota = "La interpretación experimental requiere asignación aleatoria, unidades independientes y análisis al cierre."
    p = r['p_valor_fisher_bilateral']
    p_texto = "< 0.0001" if p < .0001 else f"{p:.4g}"
    fig.supxlabel(f"Filas: {r['n_A'] + r['n_B']:,} · A: {r['n_A']:,} · B: {r['n_B']:,} · "
                  f"p-valor de Fisher bilateral: {p_texto}\n{nota}", fontsize=9)
    if args.salida:
        fig.savefig(args.salida / "comparacion_proporciones.png", dpi=150, bbox_inches="tight")
    if args.graficas:
        print("Cierra la ventana de gráficos para terminar.", flush=True)
        plt.show()
    plt.close(fig)



# 5. EJECUCIÓN: elegir comparación observacional, experimento o simulación.
def main():
    """Carga los datos, construye los grupos y presenta el análisis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=["observacional", "experimento", "demo"], default=MODO)
    parser.add_argument("--datos", "--csv", "--xlsx", dest="datos", default=RUTA_DATASET)
    parser.add_argument("--grupo", default=COLUMNA_GRUPO)
    parser.add_argument("--conversion", default=COLUMNA_CONVERSION)
    parser.add_argument("--hoja", default=HOJA_EXCEL)
    parser.add_argument("--separador", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--salida", type=Path)
    visual = parser.add_mutually_exclusive_group()
    visual.add_argument("--graficas", dest="graficas", action="store_true", help="Mostrar gráficos (predeterminado)")
    visual.add_argument("--sin-graficas", dest="graficas", action="store_false", help="No abrir ventanas; con --salida guarda PNG")
    parser.set_defaults(graficas=MOSTRAR_GRAFICAS)
    args = parser.parse_args()
    if args.modo != "demo":
        hoja = int(args.hoja) if str(args.hoja).isdigit() else args.hoja
        tabla, fuente = leer_tabla(args.datos, args.separador, hoja)
        if args.modo == "observacional":
            comparar_telco(tabla, fuente, args)
            return
        else:
            if args.grupo == args.conversion or not {args.grupo, args.conversion}.issubset(tabla.columns):
                parser.error("Para un experimento se necesitan dos columnas distintas: grupo (A/B) y resultado (0/1). Telco no las contiene.")
            tabla = tabla[[args.grupo, args.conversion]].copy()
            tabla.columns = ["grupo", "conversion"]
            etiquetas = {"A": "Grupo experimental A", "B": "Grupo experimental B"}
            print("Interpretación experimental solo si la asignación fue aleatoria y las unidades son independientes.")
    else:
        # Simulación reproducible para practicar un experimento con dos grupos.
        rng = np.random.default_rng(42)
        tabla = pd.DataFrame({"grupo": ["A"]*1000 + ["B"]*1000,
            "conversion": np.r_[rng.binomial(1, 0.12, 1000), rng.binomial(1, 0.16, 1000)]})
        fuente = "DEMOSTRACIÓN: conversiones sintéticas, semilla 42"
        etiquetas = {"A": "Grupo simulado A", "B": "Grupo simulado B"}
    resultado = analizar(tabla, args.alpha)
    # Guardar el contexto permite distinguir resultados observacionales y experimentales.
    resultado["fuente"] = fuente
    resultado["modo"] = args.modo
    resultado["grupos"] = etiquetas
    resultado["interpretacion"] = (
        "Comparación observacional y exploratoria; no demuestra causalidad."
        if args.modo == "observacional" else "Verificar asignación aleatoria e independencia para interpretación experimental.")
    print("Datos:", fuente)
    print("Filas:", len(tabla), "Grupos:", etiquetas)
    print(f"Tasa A: {resultado['tasa_A']:.2%}; tasa B: {resultado['tasa_B']:.2%}")
    print(f"Diferencia B-A: {100*resultado['diferencia_B_menos_A']:.2f} puntos porcentuales")
    intervalo = resultado['intervalo_diferencia']
    print(f"Intervalo {100*(1-args.alpha):.1f}% B-A: [{100*intervalo[0]:.2f}, {100*intervalo[1]:.2f}] puntos porcentuales")
    p = resultado['p_valor_fisher_bilateral']
    print("p-valor bilateral:", "< 0.0001" if p < .0001 else f"{p:.4g}")
    print("La prueba detecta una diferencia de proporciones bajo sus supuestos." if resultado["diferencia_significativa"] else
          "No hay evidencia suficiente de diferencia; esto no demuestra igualdad.")
    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)
        (args.salida / "resultado_ab.json").write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("Resultados guardados:", args.salida.resolve())

    if args.graficas or args.salida:
        graficar(resultado, args)


# 6. INICIO: ejecutar solo al abrir este archivo como programa.
if __name__ == "__main__":
    main()
