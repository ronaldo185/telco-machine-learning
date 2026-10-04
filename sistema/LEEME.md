# Sistema local Telco

Desde la carpeta `metodos completos`:

```powershell
py -3.12 sistema/app.py
```

Abre http://127.0.0.1:8765. Mantén la terminal abierta; Ctrl+C cierra el servidor. Si el puerto está ocupado, agrega `--port 8766`.

- Dataset: 7.043 clientes, filtros por contrato, abandono e ID; diccionario de 21 columnas y descarga del CSV original. Los filtros no cambian el entrenamiento.
- Modelos: 11 botones. Regresión predice MonthlyCharges; clasificación predice Churn (Yes=abandono). Se excluye customerID siempre y también TotalCharges/Churn de la regresión.
- Preparación: los 11 TotalCharges vacíos se imputan con la mediana aprendida solo en entrenamiento. No se borran ni inventan clientes. Los servicios sin teléfono/internet se conservan como categorías propias.
- Evaluación: 5.282 clientes de entrenamiento y 1.761 de prueba; semilla 42. Clasificación estratificada. Búsquedas de parámetros con tres particiones internas. No hay fechas para validar un pronóstico futuro.
- A/B: comparación descriptiva por contrato, sin efecto causal. La demo A/B sigue siendo una simulación separada.
- Modelo 8: ejecuta AdaBoost. LightGBM se compara cuando puede cargarse; Windows actualmente bloquea su biblioteca y aparece un aviso explícito.
- Resultados: `sistema/resultados_telco/`. Solo se conserva el historial de Telco.
- Documentos: bitácora de una página e informe de cuatro páginas en `output/pdf/`.

Los parámetros se consultan en la interfaz; se editan en cada `modelo.py`. Cada carpeta mantiene un programa independiente y una copia idéntica de `telco.csv`. La tabla web lee el CSV de la raíz. Si actualizas el archivo después, sincroniza también las copias.

Para ejecutar directamente, entra en la carpeta correspondiente y usa `py -3.12 modelo.py`; en los temas que admiten ambas tareas, añade `--tarea regresion` o `--tarea clasificacion`. Modelo 10: `--modo observacional` o `--modo demo`. `--sin-graficas` evita ventanas; `--salida resultados` guarda figuras y métricas. Se mantiene soporte para CSV y XLSX con `--datos`.

El archivo es el dataset de ejemplo IBM Telco Customer Churn; no acredita una empresa boliviana. El diccionario aportado se guardó en `datasets/telco_diccionario.json` y en la bitácora. InternetService describe el tipo de acceso, no el nombre del proveedor. Los cargos no se presentan como bolivianos.
