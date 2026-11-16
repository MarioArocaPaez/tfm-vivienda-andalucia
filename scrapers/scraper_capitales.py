"""
Scraper de CAPITALES de provincia (Fotocasa): listado + detalle en la misma pasada.

Por qué existe:
- En la extracción provincial, las capitales se dividen en muchas zonas. Cuando
  Chrome se caía a mitad de la capital, el resto de zonas fallaba en cadena con
  la sesión muerta (WinError 10061) y la capital quedaba casi vacía.
- Además, listado y detalle se hacían con semanas de diferencia y se perdía
  ~27 % de anuncios (retirados entre medias).

Qué hace:
1. Descubre las zonas (distritos) de cada capital.
2. Para cada zona: extrae el listado y, INMEDIATAMENTE, visita la ficha de
   detalle de los anuncios de esa zona (hasta un cupo por zona, para repartir
   la muestra por toda la ciudad y no concentrarla en las primeras zonas).
3. Guarda cada anuncio en el CSV en cuanto termina (no se pierde trabajo).
4. Si Chrome se cae, reinicia la sesión y reintenta la MISMA zona.
5. Se puede parar y relanzar: salta zonas completadas y anuncios ya hechos.

Reutiliza las funciones de fotocasa_base.py (listado) y
scraper_detalle_fotocasa.py (detalle y validación de identidad).

Uso (desde la carpeta scrappers):
    python scraper_capitales.py                  -> todas, en orden de prioridad
    python scraper_capitales.py malaga sevilla   -> solo las indicadas

Salida (carpeta data/capitales/):
    fotocasa_capitales_detalle.csv   mismas columnas que fotocasa_andalucia_detalle.csv
    fotocasa_capitales_listado.csv   todas las previsualizaciones vistas (con y sin detalle)
    estado_capitales.json            zonas completadas, para reanudar
    errores_capitales.csv            errores registrados

No intenta saltarse CAPTCHAs ni bloqueos: si Fotocasa limita, se registra y continúa.
"""

import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd

import fotocasa_base as fb
import scraper_detalle_fotocasa as dt


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Orden = prioridad (las que peor cobertura tenían primero).
CAPITALES = {
    "malaga":  ("Málaga",  "Málaga Capital"),
    "sevilla": ("Sevilla", "Sevilla Capital"),
    "huelva":  ("Huelva",  "Huelva Capital"),
    "cadiz":   ("Cádiz",   "Cádiz Capital"),
    "granada": ("Granada", "Granada Capital"),
    "cordoba": ("Córdoba", "Córdoba Capital"),
    "jaen":    ("Jaén",    "Jaén Capital"),
    "almeria": ("Almería", "Almería Capital"),
}
URL_CAPITAL = "https://www.fotocasa.es/es/comprar/viviendas/{slug}-capital/todas-las-zonas/l"

# Anuncios con detalle VALIDADO que queremos por capital (Aarón: ~100, mínimo 50).
# Se pide algo más para tener margen tras la limpieza. None = todos los que haya.
OBJETIVO_VALIDOS = 150

# El cupo por zona se calcula como OBJETIVO / nº zonas * este margen,
# para repartir la muestra entre distritos.
MARGEN_CUPO_ZONA = 1.5
CUPO_MINIMO_ZONA = 8

# Cuenta también los anuncios de la capital que ya están validados en el
# dataset actual y no los vuelve a visitar.
DATASET_EXISTENTE = os.path.join(fb.PROJECT_ROOT, "data", "fotocasa_andalucia_detalle.csv")
CONTAR_YA_VALIDADOS = True

MAX_REINTENTOS_ZONA = 3
REINICIAR_DRIVER_CADA = 40          # fichas de detalle entre reinicios preventivos
SEMILLA = 2026                      # muestreo reproducible dentro de cada zona

SALIDA_DIR = os.path.join(fb.PROJECT_ROOT, "data", "capitales")
CSV_DETALLE = os.path.join(SALIDA_DIR, "fotocasa_capitales_detalle.csv")
CSV_LISTADO = os.path.join(SALIDA_DIR, "fotocasa_capitales_listado.csv")
ESTADO_FILE = os.path.join(SALIDA_DIR, "estado_capitales.json")
ERRORES_FILE = os.path.join(SALIDA_DIR, "errores_capitales.csv")
CHECKPOINT_DIR = os.path.join(SALIDA_DIR, "checkpoints")

# Mismo orden de columnas que el dataset de detalle actual.
COLUMNAS = [
    "id_anuncio", "fecha_extraccion", "portal", "comunidad_autonoma", "provincia", "municipio",
    "zona", "ubicacion", "tipo_vivienda", "titulo", "precio_eur", "superficie_m2", "precio_m2",
    "habitaciones", "banos", "planta", "ascensor", "terraza", "balcon", "parking", "piscina",
    "jardin", "aire_acondicionado", "calefaccion", "trastero", "enlace",
    "fecha_primera_extraccion", "fecha_ultima_extraccion", "obra_nueva",
    "fecha_publicacion_texto", "texto_tarjeta", "archivo_origen", "url_final",
    "anuncio_disponible", "motivo_no_disponible", "titulo_detalle", "descripcion", "estado",
    "antiguedad", "antiguedad_min_anos", "antiguedad_max_anos", "amueblado", "disponibilidad",
    "cert_energia_letra", "cert_energia_consumo", "cert_energia_emision", "extras", "num_fotos",
    "latitud", "longitud", "referencia", "agencia", "orientacion", "superficie_terreno_m2",
    "bajada_precio_eur", "tiene_bajada_precio", "detalle_ok", "detalle_validado",
    "motivo_validacion",
]
COLUMNAS_LISTADO = COLUMNAS[:COLUMNAS.index("archivo_origen")]

for carpeta in (SALIDA_DIR, CHECKPOINT_DIR):
    os.makedirs(carpeta, exist_ok=True)


# ============================================================
# PERSISTENCIA
# ============================================================

def anexar_csv(ruta, fila, columnas):
    """Añade una fila al CSV (cabecera si es nuevo) y la fuerza a disco."""
    nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
    with open(ruta, "a", newline="", encoding="utf-8-sig" if nuevo else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        if nuevo:
            w.writeheader()
        w.writerow({c: fila.get(c) for c in columnas})
        f.flush()
        os.fsync(f.fileno())


def leer_csv(ruta):
    if not os.path.exists(ruta) or os.path.getsize(ruta) == 0:
        return pd.DataFrame()
    return pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)


def cargar_estado():
    if os.path.exists(ESTADO_FILE):
        with open(ESTADO_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_estado(estado):
    tmp = ESTADO_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ESTADO_FILE)


def registrar_error(capital, zona, url, error):
    anexar_csv(
        ERRORES_FILE,
        {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "capital": capital,
         "zona": zona, "url": url, "error": str(error)[:500]},
        ["fecha", "capital", "zona", "url", "error"],
    )


def ids_ya_procesados():
    """Anuncios ya visitados en ejecuciones anteriores de este script."""
    df = leer_csv(CSV_DETALLE)
    if df.empty:
        return set(), {}
    df["_k"] = df["id_anuncio"].astype(str)
    validos = df[df["detalle_validado"].astype(str).str.lower() == "true"]
    return set(df["_k"]), validos.groupby("municipio").size().to_dict()


def validados_existentes():
    """Anuncios de capitales ya validados en el dataset de detalle actual."""
    if not CONTAR_YA_VALIDADOS:
        return set(), {}
    df = leer_csv(DATASET_EXISTENTE)
    if df.empty:
        return set(), {}
    df = df[df["municipio"].astype(str).str.endswith("Capital")
            & (df["detalle_validado"].astype(str).str.lower() == "true")]
    return set(df["id_anuncio"].astype(str)), df.groupby("municipio").size().to_dict()


# ============================================================
# DRIVER COMPARTIDO
# ============================================================
# fotocasa_base usa una variable global `driver`; el scraper de detalle lo
# recibe como argumento. Mantenemos ambos apuntando a la misma sesión.

def nuevo_driver(anterior=None, motivo=None):
    d = dt.reiniciar_driver(anterior, motivo) if anterior is not None else dt.iniciar_driver_limpio()
    fb.driver = d
    return d


def driver_vivo(d):
    try:
        _ = d.current_url
        return True
    except Exception:
        return False


# ============================================================
# ZONAS
# ============================================================

PREFIJOS_ZONA = [
    "casas y pisos de compra en", "casas y pisos en venta en", "casas y pisos para comprar en",
    "venta de casas y pisos en", "pisos y casas en venta en", "viviendas en venta en",
]


def limpiar_zona(nombre):
    n = fb.normalizar_espacios(nombre or "")
    for p in PREFIJOS_ZONA:
        if n.lower().startswith(p):
            n = n[len(p):]
    return n.strip(" ,:-") or nombre


def obtener_zonas(capital, url, estado_cap):
    if estado_cap.get("zonas"):
        return estado_cap["zonas"]
    hijos, total = fb.descubrir_subzonas(capital, url)
    zonas = [{"nombre": limpiar_zona(h["nombre"]), "url": h["url"]} for h in hijos]
    if not zonas:
        zonas = [{"nombre": "Todas las zonas", "url": url}]
    estado_cap["zonas"] = zonas
    estado_cap["total_declarado"] = total
    print(f"   {len(zonas)} zonas" + (f" | Fotocasa declara ~{total:,} anuncios" if total else ""))
    return zonas


# ============================================================
# PROCESO DE UNA ZONA: listado + detalle inmediato
# ============================================================

def procesar_zona(d, provincia, capital, zona, cupo, ya_hechos, ya_validados, contador):
    """Devuelve (driver, nº validados nuevos en esta zona, zona_completada)."""
    nombre, url = zona["nombre"], zona["url"]

    # ---- 1. Listado ----
    listado = None
    for intento in range(1, MAX_REINTENTOS_ZONA + 1):
        try:
            if not driver_vivo(d):
                d = nuevo_driver(d, "sesión caída antes del listado")
            listado, _ = fb.procesar_busqueda_simple(provincia, capital, nombre, url, CHECKPOINT_DIR)
            break
        except Exception as e:
            registrar_error(capital, nombre, url, e)
            print(f"      ❌ Listado {nombre} (intento {intento}/{MAX_REINTENTOS_ZONA}): {str(e)[:120]}")
            d = nuevo_driver(d, "error en el listado")
    if listado is None:
        return d, 0, False

    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in listado:
        r.update(fecha_primera_extraccion=r.get("fecha_extraccion"),
                 fecha_ultima_extraccion=r.get("fecha_extraccion"))
        anexar_csv(CSV_LISTADO, r, COLUMNAS_LISTADO)

    # ---- 2. Detalle inmediato (muestra reproducible dentro de la zona) ----
    pendientes = [r for r in listado
                  if str(r.get("id_anuncio")) not in ya_hechos
                  and str(r.get("id_anuncio")) not in ya_validados]
    random.Random(f"{SEMILLA}|{capital}|{nombre}").shuffle(pendientes)
    if cupo is not None:
        pendientes = pendientes[:cupo]

    print(f"      🔎 Listado: {len(listado)} | detalle ahora: {len(pendientes)} (cupo {cupo})")
    validos = 0
    for i, fila in enumerate(pendientes, 1):
        if contador["desde_reinicio"] >= REINICIAR_DRIVER_CADA:
            d = nuevo_driver(d, "reinicio preventivo")
            contador["desde_reinicio"] = 0
        try:
            detalle, d = dt.extraer_detalle_con_reintentos(d, fila)
        except Exception as e:
            registrar_error(capital, nombre, fila.get("enlace"), e)
            continue
        if d is None:
            d = nuevo_driver(None)
        fb.driver = d
        contador["desde_reinicio"] += 1

        salida = dict(fila)
        salida["archivo_origen"] = "scraper_capitales"
        salida.update(detalle)
        anexar_csv(CSV_DETALLE, salida, COLUMNAS)
        ya_hechos.add(str(fila.get("id_anuncio")))

        ok = bool(detalle.get("detalle_validado"))
        validos += ok
        marca = "✓" if ok else ("✗ retirado" if detalle.get("anuncio_disponible") is False else "✗")
        print(f"         {i:3d}/{len(pendientes)} {marca} {fila.get('id_anuncio')}")
        time.sleep(random.uniform(dt.PAUSA_MIN, dt.PAUSA_MAX))

    return d, validos, True


# ============================================================
# MAIN
# ============================================================

def main(seleccion):
    estado = cargar_estado()
    ya_hechos, validos_previos_run = ids_ya_procesados()
    ya_validados, validos_dataset = validados_existentes()

    print("\n" + "#" * 62)
    print("🏙  FOTOCASA — CAPITALES (listado + detalle en la misma pasada)")
    print("#" * 62)
    print(f"Objetivo de validados por capital: {OBJETIVO_VALIDOS}")
    print(f"Salida: {CSV_DETALLE}")

    d = nuevo_driver()
    contador = {"desde_reinicio": 0}
    try:
        for slug in seleccion:
            provincia, capital = CAPITALES[slug]
            url = URL_CAPITAL.format(slug=slug)
            est = estado.setdefault(capital, {"zonas_completadas": []})

            base = validos_dataset.get(capital, 0) + validos_previos_run.get(capital, 0)
            print("\n" + "=" * 62)
            print(f"🏘 {capital} — ya validados: {base}")
            print("=" * 62)
            if OBJETIVO_VALIDOS is not None and base >= OBJETIVO_VALIDOS:
                print("   ✓ Objetivo ya alcanzado; se omite.")
                continue

            try:
                if not driver_vivo(d):
                    d = nuevo_driver(d, "sesión caída")
                zonas = obtener_zonas(capital, url, est)
                guardar_estado(estado)
            except Exception as e:
                registrar_error(capital, None, url, e)
                print(f"   ❌ No se pudieron obtener las zonas: {e}")
                d = nuevo_driver(d, "error descubriendo zonas")
                continue

            faltan = None if OBJETIVO_VALIDOS is None else OBJETIVO_VALIDOS - base
            cupo = None if faltan is None else max(
                CUPO_MINIMO_ZONA, math.ceil(faltan / len(zonas) * MARGEN_CUPO_ZONA))

            nuevos = 0
            for j, zona in enumerate(zonas, 1):
                if zona["url"] in est["zonas_completadas"]:
                    continue
                print(f"\n   🗺 {j}/{len(zonas)} {zona['nombre']}")
                d, v, completa = procesar_zona(d, provincia, capital, zona, cupo, ya_hechos, ya_validados, contador)
                nuevos += v
                if completa:
                    est["zonas_completadas"].append(zona["url"])
                est["validados"] = base + nuevos
                guardar_estado(estado)
                print(f"      📊 {capital}: {base + nuevos} validados")
                time.sleep(random.uniform(fb.PAUSA_BUSQUEDA_MIN, fb.PAUSA_BUSQUEDA_MAX))

            print(f"\n✓ {capital}: {base + nuevos} validados ({nuevos} nuevos en esta ejecución)")
            time.sleep(random.uniform(fb.PAUSA_MUNICIPIO_MIN, fb.PAUSA_MUNICIPIO_MAX))
    finally:
        dt.cerrar_driver(d)
        fb.driver = None

    resumen()


def resumen():
    df = leer_csv(CSV_DETALLE)
    if df.empty:
        return
    df["validado"] = df["detalle_validado"].astype(str).str.lower() == "true"
    print("\n" + "#" * 62)
    print("📊 RESUMEN (solo lo extraído por este script)")
    print(df.groupby("municipio").agg(visitados=("validado", "size"),
                                      validados=("validado", "sum")).to_string())


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    desconocidas = [a for a in args if a not in CAPITALES]
    if desconocidas:
        sys.exit(f"Capitales no reconocidas: {desconocidas}. Opciones: {list(CAPITALES)}")
    main(args or list(CAPITALES))
