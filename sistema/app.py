"""Interfaz local de Telco. Ejecutar: python sistema/app.py. Cerrar con Ctrl+C."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime
import argparse
import contextlib
import csv
import io
import json
import math
import mimetypes
import os
import runpy
import secrets
import subprocess
import sys
import threading
import time
import webbrowser

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATASET = ROOT / "telco.csv"
RESULTADOS = Path(os.environ.get("TELCO_RESULTADOS", str(BASE / "resultados_telco"))).resolve()
DOCUMENTOS = ROOT / "output" / "pdf"
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.RLock()
ACTIVO = None
CACHE_DATOS = None
CACHE_PARAMETROS = {}

# Los botones llaman a los mismos programas independientes que ya se utilizaban.
CATALOGO = [
    (1, "1_regresion_lineal", "Regresión lineal", ["regresion"], "Compara una recta basada en la antigüedad con un modelo que utiliza todas las entradas."),
    (2, "2_regresion_polinomial", "Regresión polinomial", ["regresion"], "Compara grado 2, Ridge, Lasso y Elastic Net con una referencia lineal."),
    (3, "3_regresion_logistica", "Regresión logística", ["clasificacion"], "Estima si el cliente abandona el servicio mediante probabilidades. Es un clasificador."),
    (4, "4_knn", "K vecinos cercanos", ["clasificacion", "regresion"], "Predice a partir de registros similares y elige K dentro del entrenamiento."),
    (5, "5_naive_bayes", "Naive Bayes", ["clasificacion"], "Compara Gaussian, Multinomial y Bernoulli con sus respectivas transformaciones."),
    (6, "6_arboles_de_decision", "Árboles de decisión", ["clasificacion", "regresion"], "Aprende reglas por divisiones. En regresión incluye Random Forest como comparación."),
    (7, "7_boosting", "Conjuntos y boosting", ["clasificacion", "regresion"], "Compara Random Forest, Gradient Boosting y AdaBoost."),
    (8, "8_adaboost", "AdaBoost y LightGBM", ["clasificacion", "regresion"], "Contrasta AdaBoost con LightGBM, dos algoritmos distintos de boosting."),
    (9, "9_gradiant_boosting", "Gradient Boosting y variantes", ["clasificacion", "regresion"], "Compara Gradient Boosting, stacking, XGBoost y CatBoost."),
    (10, "10_ab_testing", "Comparación A/B", ["observacional", "demo"], "Telco: abandono según el tipo de contrato. Demo: experimento simulado."),
    (11, "11_svm", "Máquinas de soporte vectorial", ["clasificacion", "regresion"], "Compara un modelo lineal con un kernel RBF; en regresión utiliza SVR."),
]
MODELOS = {str(i): {"id": str(i), "carpeta": folder, "nombre": name, "tareas": tasks, "descripcion": desc}
           for i, folder, name, tasks, desc in CATALOGO}
DOCS = {"bitacora_telco.pdf": "Bitácora del dataset", "informe_modelos_telco.pdf": "Informe de modelos y gráficos"}


def json_limpio(value):
    """Convierte valores de bibliotecas a JSON sin permitir NaN ni objetos ejecutables."""
    if isinstance(value, dict):
        return {str(k): json_limpio(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_limpio(v) for v in value]
    if hasattr(value, "item"):
        return json_limpio(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN (valor faltante)" if math.isnan(value) else str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def describir_estimador(est):
    """Lee parámetros reales de los objetos sklearn sin entrenarlos."""
    info = {"tipo": type(est).__name__}
    if hasattr(est, "param_grid"):
        info.update(busqueda=json_limpio(est.param_grid), criterio=est.scoring,
                    particiones=len(est.cv) if isinstance(est.cv, list) else str(est.cv),
                    estimador=describir_estimador(est.estimator))
    elif hasattr(est, "steps"):
        info["pasos"] = {name: describir_estimador(step) for name, step in est.steps if name != "preparar"}
        if any(name == "preparar" for name, _ in est.steps):
            info["preparacion"] = "Imputación, codificación del texto y escalado cuando corresponde; solo se ajustan con entrenamiento."
    else:
        params = est.get_params(deep=False)
        info["parametros"] = {k: describir_estimador(v) if hasattr(v, "get_params") else json_limpio(v)
                              for k, v in params.items() if k not in {"estimators", "tarea"}}
        if "estimators" in params:
            info["bases"] = {n: describir_estimador(m) for n, m in params["estimators"]}
    return info


def inspeccionar_parametros(modelo, tarea):
    """Consulta los objetos del modelo sin entrenarlos."""
    item = MODELOS[modelo]
    with contextlib.redirect_stdout(io.StringIO()):
        m = runpy.run_path(str(ROOT/item["carpeta"]/"modelo.py"))
        if modelo == "10":
            return {"modo":tarea, "grupos":"Mes a mes / uno o dos años" if tarea=="observacional" else "A/B simulados",
                    "inferencia":"Descriptiva, sin efecto causal" if tarea=="observacional" else "Fisher y Newcombe al 95%",
                    "dataset":"Telco: 7.043 clientes" if tarea=="observacional" else "2.000 resultados simulados, semilla 42"}
        args = argparse.Namespace(datos=str(ROOT/item["carpeta"]/m["RUTA_DATASET"]),separador=None,
                                  hoja=m["HOJA_EXCEL"],objetivo=m["COLUMNA_OBJETIVO"],tarea=tarea,
                                  columnas=m["COLUMNAS_ENTRADA"] or None)
        X,y,_ = m["cargar_datos"](args)
        train,test,_,_ = m["separar_datos"](X,y,tarea)
        estimadores=m["construir"](train,tarea)
        return {"objetivo":args.objetivo,"entradas":list(X.columns),"entrenamiento":len(train),"prueba":len(test),
                "reparto":"Clientes estratificados por Churn" if tarea=="clasificacion" else "Clientes aleatorios",
                "semilla":m["SEMILLA"],"clases":getattr(args,"clases",None),"avisos":m.get("AVISOS",[]),
                "configuracion":{k:m[k] for k in ["GRADO","C_LINEAL","C_RBF"] if k in m},
                "variantes":{n:describir_estimador(e) for n,e in estimadores.items()}}


def datos():
    """Lee CSV y diccionario; convierte TotalCharges sin rellenar sus blancos."""
    global CACHE_DATOS
    with LOCK:
        stamp=DATASET.stat().st_mtime_ns
        if CACHE_DATOS is None or CACHE_DATOS[0]!=stamp:
            import pandas as pd
            tabla=pd.read_csv(DATASET)
            tabla["TotalCharges"] = pd.to_numeric(tabla.TotalCharges.astype(str).str.strip().replace("", None), errors="coerce")
            meta=json.loads((ROOT/"datasets/telco_diccionario.json").read_text(encoding="utf-8"))
            CACHE_DATOS=(stamp,tabla,pd.DataFrame(meta["diccionario"]),pd.DataFrame(meta["ficha"]))
        return CACHE_DATOS[1:]


def guardar_job(job):
    """Guarda el estado para que los resultados sigan disponibles al reiniciar."""
    contenido = {k: v for k, v in job.items() if k != "proceso"}
    temporal = RESULTADOS / job["id"] / "estado.tmp"
    temporal.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")
    temporal.replace(temporal.with_name("estado.json"))


def ejecutar_job(job):
    global ACTIVO
    item = MODELOS[job["modelo"]]
    folder = RESULTADOS / job["id"]
    cmd = [sys.executable, "-B", "-u", str(ROOT / item["carpeta"] / "modelo.py"), "--sin-graficas", "--salida", str(folder)]
    cmd += ["--modo" if job["modelo"] == "10" else "--tarea", job["tarea"]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        with (folder / "consola.txt").open("w", encoding="utf-8") as log:
            with LOCK:
                if job["estado"] == "cancelando":
                    job["estado"] = "cancelado"
                    return
                proceso = subprocess.Popen(cmd, cwd=ROOT / item["carpeta"], stdout=log, stderr=subprocess.STDOUT,
                                           env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                job["proceso"] = proceso
            code = proceso.wait()
        with LOCK:
            job["estado"] = "cancelado" if job["estado"] == "cancelando" else ("completado" if code == 0 else "error")
            job["codigo_salida"] = code
    except Exception as ex:
        job["estado"] = "error"
        job["error"] = str(ex)
    finally:
        with LOCK:
            job["fin"] = time.time()
            guardar_job(job)
            ACTIVO = None


def iniciar_job(modelo, tarea):
    global ACTIVO
    if modelo not in MODELOS or tarea not in MODELOS[modelo]["tareas"]:
        raise ValueError("Modelo o tarea no admitidos.")
    with LOCK:
        if ACTIVO is not None:
            raise RuntimeError("Ya hay un modelo ejecutándose. Espera a que termine o cancélalo.")
        jid = datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(4)
        (RESULTADOS / jid).mkdir(parents=True)
        ACTIVO = {"id": jid, "modelo": modelo, "nombre": MODELOS[modelo]["nombre"], "tarea": tarea,
                  "estado": "ejecutando", "inicio": time.time()}
        guardar_job(ACTIVO)
        threading.Thread(target=ejecutar_job, args=(ACTIVO,), daemon=True).start()
        return jid


def leer_job(jid):
    if not jid or any(c not in "0123456789abcdef_" for c in jid):
        raise ValueError("Identificador no válido.")
    folder = RESULTADOS / jid
    with LOCK:
        if ACTIVO is not None and ACTIVO["id"] == jid:
            job = {k: v for k, v in ACTIVO.items() if k != "proceso"}
        else:
            job = json.loads((folder / "estado.json").read_text(encoding="utf-8"))
            if job["estado"] in {"ejecutando", "cancelando"}:
                job["estado"] = "interrumpido"
    log = folder / "consola.txt"
    job["log"] = log.read_text(encoding="utf-8", errors="replace")[-50000:] if log.exists() else "Preparando ejecución..."
    # Solo se ofrecen resultados completos; una cancelación puede dejar archivos parciales.
    job["archivos"] = []
    if job["estado"] == "completado":
        job["archivos"] = [{"nombre": p.name, "url": f"/resultados/{jid}/{p.name}"} for p in sorted(folder.iterdir())
                           if p.suffix in {".png", ".csv", ".json", ".txt"} and p.name != "estado.json"]
        metricas = folder / "metricas.csv"
        if metricas.exists():
            with metricas.open(encoding="utf-8", newline="") as f:
                job["metricas"] = list(csv.DictReader(f))
        contexto = folder / ("resultado_ab.json" if job["modelo"] == "10" else "ejecucion.json")
        if contexto.exists():
            job["contexto"] = json.loads(contexto.read_text(encoding="utf-8"))
    return job


class Servidor(BaseHTTPRequestHandler):
    """API pequeña, restringida a este equipo y a rutas conocidas."""
    def log_message(self, fmt, *args):
        pass

    def responder_json(self, obj, status=200):
        body = json.dumps(json_limpio(obj), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.responder(body, "application/json; charset=utf-8", status)

    def responder(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    #def archivo(self, file):
    #    self.responder(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
    def host_valido(self):
        if os.environ.get("TELCO_PUBLICO") == "1":
            return True
        return self.headers.get("Host") in {
        f"127.0.0.1:{self.server.server_port}",
        f"localhost:{self.server.server_port}"
    }

    def host_valido(self):
        return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        if not self.host_valido():
            return self.responder_json({"error": "Host no permitido."}, 403)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in {"/", "/app.js", "/style.css"}:
                return self.archivo(BASE / ("index.html" if url.path == "/" else url.path[1:]))
            if url.path == "/api/config":
                return self.responder_json({"token": TOKEN, "modelos": list(MODELOS.values()),
                    "activo": ACTIVO["id"] if ACTIVO else None,
                    "documentos": [{"nombre": name, "url": "/documentos/" + file, "disponible": (DOCUMENTOS/file).exists()} for file, name in DOCS.items()]})
            if url.path == "/api/dataset":
                tabla, dictionary, ficha = datos()
                filtrado = tabla
                contrato=query.get("contrato",[""])[0]
                abandono=query.get("churn",[""])[0]
                cliente=query.get("cliente",[""])[0].strip()
                if contrato: filtrado=filtrado.loc[filtrado.Contract.eq(contrato)]
                if abandono: filtrado=filtrado.loc[filtrado.Churn.eq(abandono)]
                if cliente: filtrado=filtrado.loc[filtrado.customerID.str.contains(cliente,case=False,regex=False)]
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(100, max(1, int(query.get("limit", [30])[0])))
                numeric = tabla.select_dtypes(include="number").describe().round(3).fillna(0)
                return self.responder_json({"total": len(tabla), "filtrados": len(filtrado), "offset": offset, "limit": limit,
                    "columnas": list(tabla.columns), "filas": filtrado.iloc[offset:offset+limit].astype(object).where(filtrado.iloc[offset:offset+limit].notna(),None).to_dict("records"),
                    "contratos": sorted(tabla.Contract.unique()), "columnas_total": len(tabla.columns),
                    "vacios": int(tabla.isna().sum().sum()), "abandonos": int(tabla.Churn.eq("Yes").sum()),
                    "duplicados": int(tabla.customerID.duplicated().sum()),
                    "diccionario": dictionary.to_dict("records"), "ficha": ficha.to_dict("records"),
                    "estadisticas": numeric.reset_index().rename(columns={"index": "medida"}).to_dict("records")})
            if url.path == "/api/parametros":
                modelo, tarea = query.get("modelo", [""])[0], query.get("tarea", [""])[0]
                if modelo not in MODELOS or tarea not in MODELOS[modelo]["tareas"]: raise ValueError("Selección no válida.")
                folder = ROOT/MODELOS[modelo]["carpeta"]
                key = (modelo, tarea, (folder/"modelo.py").stat().st_mtime_ns, (folder/"telco.csv").stat().st_mtime_ns)
                if key not in CACHE_PARAMETROS:
                    p = subprocess.run([sys.executable, "-B", str(BASE/"app.py"), "--parametros", modelo, "--tarea", tarea],
                                       capture_output=True, text=True, encoding="utf-8", timeout=90,
                                       env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if p.returncode: raise ValueError(p.stderr[-3000:] or p.stdout[-3000:])
                    CACHE_PARAMETROS[key] = json.loads(p.stdout)
                return self.responder_json(CACHE_PARAMETROS[key])
            if url.path == "/api/jobs":
                jobs=[]
                for p in sorted(RESULTADOS.glob("*/estado.json"), reverse=True)[:20]:
                    job=leer_job(p.parent.name)
                    jobs.append({k:job[k] for k in ["id","modelo","nombre","tarea","estado","inicio"]})
                return self.responder_json(jobs)
            if url.path.startswith("/api/jobs/"):
                return self.responder_json(leer_job(url.path.rsplit("/",1)[1]))
            if url.path.startswith("/documentos/"):
                name = url.path.rsplit("/",1)[1]
                if name not in DOCS: raise FileNotFoundError()
                return self.archivo(DOCUMENTOS/name)
            if url.path.startswith("/resultados/"):
                parts = unquote(url.path).split("/")
                if len(parts)!=4: raise ValueError("Ruta no válida.")
                job=leer_job(parts[2])
                if parts[3] not in {x["nombre"] for x in job["archivos"]}: raise FileNotFoundError()
                return self.archivo(RESULTADOS/parts[2]/parts[3])
            if url.path == "/dataset.csv":
                return self.archivo(DATASET)
            self.responder_json({"error":"Ruta no encontrada."},404)
        except FileNotFoundError:
            self.responder_json({"error":"Archivo o ejecución no encontrados."},404)
        except (ValueError, KeyError, subprocess.TimeoutExpired) as ex:
            self.responder_json({"error":str(ex)},400)
        except Exception as ex:
            self.responder_json({"error":f"No se pudo completar la operación: {ex}"},500)

    def do_POST(self):
        if not self.host_valido() or not secrets.compare_digest(self.headers.get("X-Local-Token", ""), TOKEN):
            return self.responder_json({"error":"Solicitud local no autorizada. Recarga la página."},403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size<1 or size>4096: raise ValueError("Tamaño de solicitud no permitido.")
            body = json.loads(self.rfile.read(size))
            if self.path == "/api/run":
                jid = iniciar_job(str(body.get("modelo", "")), body.get("tarea", ""))
                return self.responder_json({"id":jid},202)
            if self.path == "/api/cancel":
                with LOCK:
                    if not ACTIVO or ACTIVO["id"] != body.get("id"):
                        raise ValueError("La ejecución ya terminó o no existe.")
                    ACTIVO["estado"]="cancelando"
                    if ACTIVO.get("proceso"): ACTIVO["proceso"].terminate()
                return self.responder_json({"estado":"cancelando"})
            self.responder_json({"error":"Ruta no encontrada."},404)
        except RuntimeError as ex:
            self.responder_json({"error":str(ex)},409)
        except (ValueError, TypeError, AttributeError) as ex:
            self.responder_json({"error":str(ex)},400)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--no-browser",action="store_true")
    parser.add_argument("--parametros",choices=list(MODELOS),help=argparse.SUPPRESS)
    parser.add_argument("--tarea",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.parametros:
        print(json.dumps(inspeccionar_parametros(args.parametros,args.tarea),ensure_ascii=False,allow_nan=False))
        return
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    try:
        #server=ThreadingHTTPServer(("127.0.0.1",args.port),Servidor)
        server=ThreadingHTTPServer(("0.0.0.0",args.port),Servidor)
    except OSError as ex:
        raise SystemExit(f"No se pudo abrir el puerto {args.port}. Prueba --port 8766. Detalle: {ex}")
    url=f"http://127.0.0.1:{server.server_port}"
    print(f"Sistema Telco: {url}\nMantén esta terminal abierta. Para cerrar: Ctrl+C.",flush=True)
    if not args.no_browser:
        threading.Timer(.6,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            if ACTIVO and ACTIVO.get("proceso"):
                ACTIVO["estado"]="cancelando"
                ACTIVO["proceso"].terminate()
        server.server_close()


if __name__=="__main__":
    main()
