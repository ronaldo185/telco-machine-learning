import os
import glob
import json
from datetime import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
CARPETA_VERSIONES = "D:/ml/exp2/metodos completos/metodos completos/versionanterior"
RUTA_NUEVO = "D:/ml/exp2/metodos completos/metodos completos/telco.csv"
SALIDA_PDF = "D:/ml/exp2/metodos completos/metodos completos/output/pdf/bitacora_telco.pdf"
RUTA_HISTORIAL_JSON = "D:/ml/exp2/metodos completos/metodos completos/output/pdf/historial_bitacora.json"

# ==============================================================================
# DICCIONARIO DE DATOS DE TELCO CHURN
# ==============================================================================
DICCIONARIO_COLUMNAS = [
    ("customerID", "Texto / String", "ID único e identificador de cada cliente"),
    ("gender", "Texto", "Género del cliente (Male / Female)"),
    ("SeniorCitizen", "Entero / Int", "Indica si es adulto mayor (1) o no (0)"),
    ("Partner", "Texto", "Indica si el cliente tiene pareja (Yes / No)"),
    ("Dependents", "Texto", "Indica si tiene dependientes económicos (Yes / No)"),
    ("tenure", "Entero / Int", "Meses de antigüedad del cliente en la empresa"),
    ("PhoneService", "Texto", "Servicio de telefonía contratado (Yes / No)"),
    ("MultipleLines", "Texto", "Líneas telefónicas múltiples (Yes / No / No phone service)"),
    ("InternetService", "Texto", "Proveedor de internet (DSL / Fiber optic / No)"),
    ("OnlineSecurity", "Texto", "Seguridad en línea (Yes / No / No internet service)"),
    ("OnlineBackup", "Texto", "Respaldos en la nube (Yes / No / No internet service)"),
    ("DeviceProtection", "Texto", "Protección de dispositivos (Yes / No / No internet service)"),
    ("TechSupport", "Texto", "Soporte técnico prioritario (Yes / No / No internet service)"),
    ("StreamingTV", "Texto", "Servicio de TV por streaming (Yes / No / No internet service)"),
    ("StreamingMovies", "Texto", "Servicio de películas por streaming (Yes / No / No internet service)"),
    ("Contract", "Texto", "Tipo de contrato (Month-to-month / One year / Two year)"),
    ("PaperlessBilling", "Texto", "Facturación electrónica activada (Yes / No)"),
    ("PaymentMethod", "Texto", "Método de pago (Electronic check / Mailed check / etc.)"),
    ("MonthlyCharges", "Flotante / Float", "Monto cobrado mensualmente al cliente"),
    ("TotalCharges", "Flotante / Float", "Monto total acumulado gastado por el cliente"),
    ("Churn", "Texto", "Variable objetivo: Si el cliente abandonó la empresa (Yes / No)")
]

# ==============================================================================
# LÓGICA DE BÚSQUEDA DINÁMICA Y ANÁLISIS
# ==============================================================================
def obtener_ultimo_dataset_antiguo(directorio):
    """
    Busca los archivos que coincidan con 'telco_version_*.csv' en la carpeta indicada
    y selecciona el más reciente basándose en su fecha de modificación.
    """
    patron = os.path.join(directorio, "telco_version_*.csv")
    archivos = glob.glob(patron)
    
    if not archivos:
        return None
    
    # Devuelve la ruta del archivo modificado/creado más recientemente
    archivo_mas_reciente = max(archivos, key=os.path.getmtime)
    return archivo_mas_reciente

def analizar_cambios(ruta_original, ruta_nuevo):
    """Compara dos datasets y calcula las métricas de diferencias."""
    df_orig = pd.read_csv(ruta_original)
    df_nuevo = pd.read_csv(ruta_nuevo)

    filas_orig, cols_orig = df_orig.shape
    filas_nuevo, cols_nuevo = df_nuevo.shape

    cols_agregadas = list(set(df_nuevo.columns) - set(df_orig.columns))
    cols_eliminadas = list(set(df_orig.columns) - set(df_nuevo.columns))

    nulos_orig = int(df_orig.isna().sum().sum())
    nulos_nuevo = int(df_nuevo.isna().sum().sum())
    duplicados_orig = int(df_orig.duplicated().sum())
    duplicados_nuevo = int(df_nuevo.duplicated().sum())

    return {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_anterior": os.path.basename(ruta_original),
        "filas_orig": filas_orig,
        "filas_nuevo": filas_nuevo,
        "diff_filas": filas_nuevo - filas_orig,
        "cols_orig": cols_orig,
        "cols_nuevo": cols_nuevo,
        "cols_agregadas": ", ".join(cols_agregadas) if cols_agregadas else "Ninguna",
        "cols_eliminadas": ", ".join(cols_eliminadas) if cols_eliminadas else "Ninguna",
        "nulos_orig": nulos_orig,
        "nulos_nuevo": nulos_nuevo,
        "duplicados_orig": duplicados_orig,
        "duplicados_nuevo": duplicados_nuevo,
    }

def guardar_historial_json(nuevo_registro, ruta_json):
    """Guarda el historial acumulado en formato JSON sin sobrescribir el pasado."""
    os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
    historial = []
    
    if os.path.exists(ruta_json):
        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                historial = json.load(f)
        except Exception:
            historial = []

    historial.append(nuevo_registro)

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    return historial

# ==============================================================================
# GENERACIÓN DE PDF BITÁCORA
# ==============================================================================
def generar_pdf_bitacora(historial_completo, ruta_pdf_salida):
    """Genera el reporte PDF con el diccionario en la pág. 1 e historial acumulativo a continuación."""
    os.makedirs(os.path.dirname(ruta_pdf_salida), exist_ok=True)
    doc = SimpleDocTemplate(
        ruta_pdf_salida,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=18, leading=22,
        textColor=colors.HexColor("#0f172a"), spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontSize=9, leading=12,
        textColor=colors.HexColor("#475569"), spaceAfter=10
    )
    heading2_style = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'], fontSize=12, leading=15,
        textColor=colors.HexColor("#1e293b"), spaceBefore=10, spaceAfter=6
    )
    cell_style = ParagraphStyle(
        'TableCell', parent=styles['Normal'], fontSize=8, leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    cell_bold = ParagraphStyle(
        'TableCellBold', parent=styles['Normal'], fontSize=8, leading=10,
        textColor=colors.HexColor("#0f172a"), fontName='Helvetica-Bold'
    )

    elements = []

    # --------------------------------------------------------------------------
    # PÁGINA 1: DICCIONARIO DE DATOS
    # --------------------------------------------------------------------------
    elements.append(Paragraph("Bitácora de Control y Diccionario de Datos", title_style))
    elements.append(Paragraph("<b>Estructura general y especificación de campos del dataset</b>", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    elements.append(Paragraph("1. Diccionario de Datos", heading2_style))

    tabla_dicc_data = [[
        Paragraph("Nombre Variable", cell_bold),
        Paragraph("Tipo de Dato", cell_bold),
        Paragraph("Descripción", cell_bold)
    ]]

    for col_nombre, col_tipo, col_desc in DICCIONARIO_COLUMNAS:
        tabla_dicc_data.append([
            Paragraph(col_nombre, cell_bold),
            Paragraph(col_tipo, cell_style),
            Paragraph(col_desc, cell_style)
        ])

    t_dicc = Table(tabla_dicc_data, colWidths=[120, 100, 320])
    t_dicc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_dicc)

    # Salto de página para separar el diccionario del historial de cambios
    elements.append(PageBreak())

    # --------------------------------------------------------------------------
    # PÁGINAS SIGUIENTES: HISTORIAL DE CONTROL DE CAMBIOS
    # --------------------------------------------------------------------------
    elements.append(Paragraph("2. Historial de Control de Cambios", title_style))
    elements.append(Paragraph(f"<b>Total de eventos registrados:</b> {len(historial_completo)} | <b>Registro continuo acumulado</b>", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    for idx, datos in enumerate(reversed(historial_completo), 1):
        num_evento = len(historial_completo) - idx + 1
        archivo_ref = datos.get("archivo_anterior", "Dataset Anterior")
        
        elements.append(Paragraph(f"Evento #{num_evento} - Fecha: {datos['fecha']} (Base: {archivo_ref})", heading2_style))
        
        # Tabla de Dimensiones
        tabla_dim = [
            ["Métrica", "Dataset Original", "Dataset Modificado", "Diferencia"],
            ["Total Registros (Filas)", str(datos["filas_orig"]), str(datos["filas_nuevo"]), f"{datos['diff_filas']:+}"],
            ["Total Columnas", str(datos["cols_orig"]), str(datos["cols_nuevo"]), f"{datos['cols_nuevo'] - datos['cols_orig']:+}"]
        ]
        t_dim = Table(tabla_dim, colWidths=[180, 110, 110, 110])
        t_dim.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]))
        elements.append(t_dim)
        elements.append(Spacer(1, 6))

        # Tabla de Calidad y Estructura
        tabla_cal = [
            ["Indicador de Calidad", "Original", "Modificado", "Columnas Añadidas / Eliminadas"],
            ["Valores Nulos / Faltantes", str(datos["nulos_orig"]), str(datos["nulos_nuevo"]), f"+ {datos['cols_agregadas']}"],
            ["Registros Duplicados", str(datos["duplicados_orig"]), str(datos["duplicados_nuevo"]), f"- {datos['cols_eliminadas']}"]
        ]
        t_cal = Table(tabla_cal, colWidths=[150, 80, 80, 200])
        t_cal.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('ALIGN', (1,0), (2,-1), 'CENTER'),
        ]))
        elements.append(t_cal)
        elements.append(Spacer(1, 14))

    doc.build(elements)
    print(f"Bitácora PDF generada con éxito en: {ruta_pdf_salida}")

# ==============================================================================
# BLOQUE PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # Buscar dinámicamente la versión antigua más reciente en la carpeta
    RUTA_ORIGINAL = obtener_ultimo_dataset_antiguo(CARPETA_VERSIONES)

    if RUTA_ORIGINAL and os.path.exists(RUTA_ORIGINAL) and os.path.exists(RUTA_NUEVO):
        print(f"Versión anterior detectada dinámicamente: {RUTA_ORIGINAL}")
        nuevo_evento = analizar_cambios(RUTA_ORIGINAL, RUTA_NUEVO)
        historial_actualizado = guardar_historial_json(nuevo_evento, RUTA_HISTORIAL_JSON)
        generar_pdf_bitacora(historial_actualizado, SALIDA_PDF)
    else:
        if not RUTA_ORIGINAL:
            print(f"No se encontró ningún archivo con el patrón 'telco_version_*.csv' en: {CARPETA_VERSIONES}")
        elif not os.path.exists(RUTA_NUEVO):
            print(f"No se encontró el dataset nuevo en: {RUTA_NUEVO}")