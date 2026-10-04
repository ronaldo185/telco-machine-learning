import os
import shutil
import numpy as np
import pandas as pd

# 1. Definición de rutas principales
directorio_base = r"D:/ml/exp2/metodos completos/metodos completos"
ruta_original = os.path.join(directorio_base, "telco.csv")
carpeta_backup = os.path.join(directorio_base, "versionanterior")

# LISTA DE LAS 11 CARPETAS/DIRECTORIOS DE DESTINO
directorios_destino = [
    r"D:/ml/exp2/metodos completos/metodos completos/1_regresion_lineal",
    r"D:/ml/exp2/metodos completos/metodos completos/2_regresion_polinomial",
    r"D:/ml/exp2/metodos completos/metodos completos/3_regresion_logistica",
    r"D:/ml/exp2/metodos completos/metodos completos/4_knn",
    r"D:/ml/exp2/metodos completos/metodos completos/5_naive_bayes",
    r"D:/ml/exp2/metodos completos/metodos completos/6_arboles_de_decision",
    r"D:/ml/exp2/metodos completos/metodos completos/7_boosting",
    r"D:/ml/exp2/metodos completos/metodos completos/8_adaboost",
    r"D:/ml/exp2/metodos completos/metodos completos/9_gradiant_boosting",
    r"D:/ml/exp2/metodos completos/metodos completos/10_ab_testing",
    r"D:/ml/exp2/metodos completos/metodos completos/11_svm",
]

os.makedirs(carpeta_backup, exist_ok=True)

# 2. Cargar el dataset actual
if not os.path.exists(ruta_original):
    raise FileNotFoundError(
        f"No se encontró el archivo base en: {ruta_original}"
    )

df_existente = pd.read_csv(ruta_original, low_memory=False)
df_existente["TotalCharges"] = pd.to_numeric(
    df_existente["TotalCharges"], errors="coerce"
)
df_existente = df_existente.dropna(subset=["TotalCharges"]).reset_index(
    drop=True
)

# 3. MOVER EL ARCHIVO ANTERIOR A 'versionanterior'
timestamp_backup = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
nombre_backup = f"telco_version_{timestamp_backup}.csv"
ruta_backup = os.path.join(carpeta_backup, nombre_backup)

shutil.move(ruta_original, ruta_backup)
print(f"Backup del dataset previo guardado en: {ruta_backup}")

# 4. Generar ÚNICAMENTE los registros de 1 solo día
np.random.seed()  # Semilla aleatoria variante por ejecución
media_diaria = 1000
desviacion_diaria = 35
cantidad_hoy = int(np.random.normal(media_diaria, desviacion_diaria))

# Muestrear clientes existentes para mantener distribuciones del perfil original
df_hoy = df_existente.sample(n=cantidad_hoy, replace=True).copy()

# Asignar nuevos IDs únicos
total_filas_previas = len(df_existente)
df_hoy["customerID"] = [
    f"NEW-{total_filas_previas + i + 1:05d}" for i in range(cantidad_hoy)
]

# CORRECCIONES DE COHERENCIA DE NEGOCIO:
df_hoy["tenure"] = 1
df_hoy["Churn"] = "No"

variacion = np.random.uniform(0.97, 1.03, size=cantidad_hoy)
df_hoy["MonthlyCharges"] = (df_hoy["MonthlyCharges"] * variacion).round(2)
df_hoy["TotalCharges"] = df_hoy["MonthlyCharges"]

# SOLUCIÓN AL ERROR '.str accessor':
# Asegurar que todas las columnas de texto en los datos nuevos sean puramente cadenas de texto sin NaNs
columnas_texto = df_hoy.select_dtypes(include=["object"]).columns
for col in columnas_texto:
    df_hoy[col] = df_hoy[col].fillna("No").astype(str)

# 5. Unir registros pasados + registros sintéticos de hoy
df_actualizado = pd.concat([df_existente, df_hoy], ignore_index=True)

# Guardar en la ruta principal (telco.csv)
df_actualizado.to_csv(ruta_original, index=False)

# 6. COPIAR EL DATASET ACTUALIZADO A LOS 11 DIRECTORIOS
print("\nCopiando dataset a los directorios de destino...")
for carpeta in directorios_destino:
    os.makedirs(carpeta, exist_ok=True)  # Crea la carpeta si no existe
    ruta_copia = os.path.join(carpeta, "telco.csv")
    shutil.copy2(ruta_original, ruta_copia)
    print(f"  ➜ Copiado en: {ruta_copia}")

print("\n" + "=" * 60)
print("REGISTRO SINTÉTICO GENERADO Y REPLICADO EXITOSAMENTE")
print("=" * 60)
print(f"• Clientes añadidos hoy: {cantidad_hoy}")
print(f"• Total acumulado en telco.csv: {len(df_actualizado)} filas.")
print(f"• Replicado en: {len(directorios_destino)} directorios distintas.")

import subprocess

# 7. EJECUTAR OTRO PROGRAMA O SCRIPT DE PYTHON
#D:/ml/exp2/metodos completos/metodos completos/output/pdf
script_a_ejecutar = r"D:/ml/exp2/metodos completos/metodos completos/bitacora.py"

print("\n🚀 Ejecutando el siguiente programa...")
subprocess.run(["python", script_a_ejecutar], check=True)