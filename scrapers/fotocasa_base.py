"""
Scraper de listados de Fotocasa para las provincias de Andalucía.

Características principales:
- Descubre municipios y zonas desde las páginas provinciales.
- Acumula anuncios durante el scroll para soportar DOM virtualizado.
- Deduplica por identificador de anuncio y, como alternativa, por URL.
- Guarda checkpoints acumulativos para poder reanudar ejecuciones.
- Reescanea búsquedas ya procesadas para capturar anuncios nuevos.
- Conserva snapshots de cada ejecución.
- Registra errores y estado por provincia.
- Incluye anuncios de obra nueva cuando aparecen en los listados.
- Genera un diagnóstico de cobertura comparando resultados anunciados
  por Fotocasa con anuncios únicos observados.

El código no intenta eludir CAPTCHAs, bloqueos ni otras medidas de acceso.
Si el portal limita una búsqueda, el error se registra y la ejecución continúa.
"""

import gc
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# CONFIGURACIÓN
# ============================================================

COMUNIDAD_AUTONOMA = "Andalucía"
HEADLESS = False

PAGE_LOAD_TIMEOUT = 60
WAIT_ANUNCIOS_TIMEOUT = 18

# Scroll incremental. Más conservador que saltar directamente al final.
SCROLL_STEP_RATIO = 0.85
SCROLL_WAIT_MIN = 1.4
SCROLL_WAIT_MAX = 2.4
MAX_SCROLLS = 450
MAX_RONDAS_SIN_NUEVOS = 10

# Si una búsqueda supera este tamaño intentamos dividirla geográficamente.
# Lo bajamos respecto al original para reducir el riesgo de topes del listado.
UMBRAL_DIVIDIR_ZONAS = 350
MAX_NIVEL_ZONAS = 2

PAUSA_BUSQUEDA_MIN = 2.0
PAUSA_BUSQUEDA_MAX = 4.0
PAUSA_MUNICIPIO_MIN = 4.0
PAUSA_MUNICIPIO_MAX = 8.0

MUNICIPIOS_POR_DRIVER = 5
MAX_REINTENTOS_DRIVER = 4

# Reescanea también lo que el estado antiguo marcó como completado.
# Esto es importante si quieres recuperar anuncios que la pasada anterior no vio.
REESCANEAR_COMPLETADAS = True

# Refresca la lista de municipios y la fusiona con la guardada anteriormente.
REFRESCAR_MUNICIPIOS = True

# Si una búsqueda grande no se puede subdividir, se procesa igualmente.
PROCESAR_PADRE_AUNQUE_HAYA_HIJOS = False

# Fotocasa actualmente funciona principalmente con carga progresiva, pero
# mantenemos soporte de enlaces "next" si aparecen.
MAX_PAGINAS_POR_BUSQUEDA = 250


driver = None


# ============================================================
# RUTAS
# ============================================================

SCRAPPERS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRAPPERS_DIR)

# Los datos no forman parte del repositorio. La carpeta se crea al ejecutar
# el scraper y queda excluida por .gitignore.
DATA_PROVINCIAS_DIR = os.path.join(PROJECT_ROOT, "data", "provincias")
CHECKPOINTS_ROOT = os.path.join(DATA_PROVINCIAS_DIR, "checkpoints")
ESTADOS_ROOT = os.path.join(DATA_PROVINCIAS_DIR, "estados")
ERRORES_ROOT = os.path.join(DATA_PROVINCIAS_DIR, "errores")
RUNS_ROOT = os.path.join(DATA_PROVINCIAS_DIR, "runs")

for carpeta in [DATA_PROVINCIAS_DIR, CHECKPOINTS_ROOT, ESTADOS_ROOT, ERRORES_ROOT, RUNS_ROOT]:
    os.makedirs(carpeta, exist_ok=True)


# ============================================================
# UTILIDADES
# ============================================================

def slugify(texto):
    if not texto:
        return "sin-nombre"
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]+", "-", texto)
    return texto.strip("-")


def normalizar_espacios(texto):
    return re.sub(r"\s+", " ", texto or "").strip()


def normalizar_url_navegacion(href):
    """Conserva query params. Solo elimina fragmentos (#...)."""
    if not href:
        return None
    url = urljoin("https://www.fotocasa.es", href)
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))


def normalizar_url_anuncio(href):
    """Para deduplicar anuncios sí quitamos query/fragment, no el path."""
    if not href:
        return None
    url = urljoin("https://www.fotocasa.es", href)
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))


def clave_anuncio(registro):
    valor_id = registro.get("id_anuncio")
    if valor_id is not None and str(valor_id).strip() not in ("", "nan", "None"):
        return f"id:{str(valor_id).strip()}"
    enlace = registro.get("enlace")
    if enlace:
        return f"url:{normalizar_url_anuncio(str(enlace))}"
    # Último recurso: evita perder la fila aunque no tenga enlace.
    return "fallback:" + "|".join(
        str(registro.get(k, ""))
        for k in ["titulo", "precio_eur", "superficie_m2", "ubicacion"]
    )


def obtener_rutas_provincia(provincia):
    s = slugify(provincia)
    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, s)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return {
        "checkpoint_dir": checkpoint_dir,
        "estado_file": os.path.join(ESTADOS_ROOT, f"{s}.json"),
        "errores_file": os.path.join(ERRORES_ROOT, f"{s}.csv"),
        "output_file": os.path.join(DATA_PROVINCIAS_DIR, f"fotocasa_{s}.csv"),
        "runs_dir": os.path.join(RUNS_ROOT, s),
    }


# ============================================================
# DRIVER
# ============================================================

def crear_driver():
    options = webdriver.ChromeOptions()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-features=BackForwardCache")
    options.page_load_strategy = "eager"
    options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
        },
    )
    d = webdriver.Chrome(options=options)
    d.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return d


def cerrar_driver():
    global driver
    if driver is not None:
        try:
            driver.quit()
        except Exception:
            pass
    driver = None
    gc.collect()


def reiniciar_driver():
    global driver
    cerrar_driver()
    ultimo_error = None
    for intento in range(1, MAX_REINTENTOS_DRIVER + 1):
        try:
            if intento > 1:
                time.sleep(intento * 4)
            print(f"\n🌐 Iniciando Chrome ({intento}/{MAX_REINTENTOS_DRIVER})...")
            driver = crear_driver()
            return driver
        except Exception as e:
            ultimo_error = e
            driver = None
            gc.collect()
    raise RuntimeError(f"No se pudo iniciar Chrome: {ultimo_error}")


def safe_get(url):
    global driver
    if driver is None:
        return False
    try:
        driver.get(url)
        return True
    except TimeoutException:
        print(f"⚠ Timeout cargando {url}; se intenta continuar con lo recibido.")
        try:
            driver.execute_script("window.stop();")
            return True
        except Exception:
            return False
    except WebDriverException as e:
        print(f"❌ Error navegando a {url}: {e}")
        return False


def aceptar_cookies():
    try:
        btn = WebDriverWait(driver, 6).until(
            EC.element_to_be_clickable((By.ID, "didomi-notice-agree-button"))
        )
        btn.click()
        time.sleep(0.5)
    except Exception:
        pass


def esperar_anuncios():
    try:
        WebDriverWait(driver, WAIT_ANUNCIOS_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
        )
        return True
    except Exception:
        return False


# ============================================================
# ESTADO / ERRORES
# ============================================================

def cargar_estado(path):
    if not os.path.exists(path):
        return {"busquedas_completadas": [], "municipios_descubiertos": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            estado = json.load(f)
        estado.setdefault("busquedas_completadas", [])
        estado.setdefault("municipios_descubiertos", [])
        return estado
    except Exception:
        return {"busquedas_completadas": [], "municipios_descubiertos": []}


def guardar_estado(path, estado):
    estado["ultima_actualizacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)


def registrar_error(path, provincia, municipio, zona, url, error):
    fila = pd.DataFrame(
        [{
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "provincia": provincia,
            "municipio": municipio,
            "zona": zona,
            "url": url,
            "error": str(error),
        }]
    )
    fila.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# PARSEO DE RESULTADOS
# ============================================================

def obtener_total_resultados_desde_soup(soup):
    h1 = soup.find("h1")
    if not h1:
        return None
    texto = normalizar_espacios(h1.get_text(" ", strip=True))
    # Ej.: 9.879 Casas y pisos ...
    m = re.search(r"(\d[\d\.\s]*)\s+(?:casas|viviendas|pisos|inmuebles)", texto, re.I)
    if not m:
        return None
    try:
        return int(re.sub(r"\D", "", m.group(1)))
    except ValueError:
        return None


def obtener_total_resultados():
    try:
        return obtener_total_resultados_desde_soup(BeautifulSoup(driver.page_source, "html.parser"))
    except Exception:
        return None


def extraer_numero_es(texto):
    if not texto:
        return None
    s = texto.strip().replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # En precios/superficies españolas el punto suele ser separador de miles.
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def extraer_precio(texto):
    m = re.search(r"(\d[\d\.\,]*)\s*€", texto or "")
    return extraer_numero_es(m.group(1)) if m else None


def extraer_superficie(texto):
    m = re.search(r"(\d[\d\.\,]*)\s*m(?:²|2)\b", texto or "", re.I)
    return extraer_numero_es(m.group(1)) if m else None


ERRORES_SESION_MUERTA = (
    "failed to establish a new connection", "max retries exceeded", "connection refused",
    "winerror 10061", "winerror 10054", "invalid session id", "chrome not reachable",
    "session deleted", "no such window", "disconnected",
)


def es_sesion_muerta(error):
    """True si Chrome/ChromeDriver ha caído y la sesión ya no sirve."""
    texto = str(error).lower()
    return any(p in texto for p in ERRORES_SESION_MUERTA)


def es_href_anuncio(href):
    if not href:
        return False
    h = href.lower()
    # La versión anterior descartaba explícitamente obra nueva.
    return (
        "/comprar/vivienda/" in h
        or "/obra-nueva/" in h
    )


def extraer_id_anuncio(enlace):
    if not enlace:
        return None
    # IDs de Fotocasa suelen ser secuencias largas; evitamos capturar años/metros.
    candidatos = re.findall(r"(?<!\d)(\d{6,})(?!\d)", urlparse(enlace).path)
    return candidatos[-1] if candidatos else None


def extraer_tipo_vivienda(titulo):
    t = (titulo or "").lower()
    tipos = [
        "planta baja", "piso", "apartamento", "ático", "atico", "casa", "chalet",
        "dúplex", "duplex", "estudio", "adosado", "pareado", "finca", "villa",
    ]
    for tipo in tipos:
        if tipo in t:
            return tipo.replace("atico", "ático").replace("duplex", "dúplex")
    return None


def extraer_anuncios(soup, provincia, municipio, zona):
    resultados = []
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for article in soup.find_all("article"):
        try:
            if article.find(attrs={"data-panot-component": "skeleton"}):
                continue

            enlace = None
            obra_nueva = False
            for a in article.find_all("a", href=True):
                href = a.get("href")
                if es_href_anuncio(href):
                    enlace = normalizar_url_anuncio(href)
                    obra_nueva = "/obra-nueva/" in href.lower()
                    break
            if not enlace:
                continue

            h3 = article.find("h3")
            titulo = normalizar_espacios(h3.get_text(" ", strip=True)) if h3 else None
            texto = normalizar_espacios(article.get_text(" ", strip=True))
            tl = texto.lower()

            precio = extraer_precio(texto)
            superficie = extraer_superficie(texto)

            mh = re.search(r"(\d+)\s*habs?\b", tl)
            mb = re.search(r"(\d+)\s*baños?\b", tl)
            habitaciones = int(mh.group(1)) if mh else None
            banos = int(mb.group(1)) if mb else None

            precio_m2 = round(precio / superficie, 2) if precio and superficie and superficie > 0 else None

            planta = None
            mp = re.search(
                r"\b(bajos?|entresuelo|semisótano|sótano|\d+\s*[ªº]?\s*planta)\b",
                tl,
                re.I,
            )
            if mp:
                planta = normalizar_espacios(mp.group(1))

            # Texto temporal mostrado en tarjeta: "Ayer", "Hace 5 días", etc.
            fecha_publicacion_texto = None
            mf = re.search(r"\b(Ayer|Hoy|Hace\s+\d+\s+días?|Más de\s+\d+\s+mes(?:es)?)\b", texto, re.I)
            if mf:
                fecha_publicacion_texto = mf.group(1)

            # Ubicación: preferimos textos cortos que mencionen municipio/provincia.
            ubicacion = None
            municipio_simple = (municipio or "").lower().replace(" capital", "")
            candidatos = []
            for el in article.find_all(["span", "p", "div"]):
                txt = normalizar_espacios(el.get_text(" ", strip=True))
                if 2 <= len(txt) <= 150:
                    candidatos.append(txt)
            geo = [
                x for x in candidatos
                if municipio_simple in x.lower() or provincia.lower() in x.lower()
            ]
            if geo:
                ubicacion = min(geo, key=len)

            resultados.append({
                "id_anuncio": extraer_id_anuncio(enlace),
                "fecha_extraccion": fecha,
                "portal": "Fotocasa",
                "comunidad_autonoma": COMUNIDAD_AUTONOMA,
                "provincia": provincia,
                "municipio": municipio,
                "zona": zona,
                "ubicacion": ubicacion,
                "tipo_vivienda": extraer_tipo_vivienda(titulo),
                "obra_nueva": obra_nueva,
                "titulo": titulo,
                "precio_eur": precio,
                "superficie_m2": superficie,
                "precio_m2": precio_m2,
                "habitaciones": habitaciones,
                "banos": banos,
                "planta": planta,
                "ascensor": "ascensor" in tl,
                "terraza": "terraza" in tl,
                "balcon": "balcón" in tl or "balcon" in tl,
                "parking": "parking" in tl or "garaje" in tl,
                "piscina": "piscina" in tl,
                "jardin": "jardín" in tl or "jardin" in tl,
                "aire_acondicionado": "aire acondicionado" in tl,
                "calefaccion": "calefacción" in tl or "calefaccion" in tl,
                "trastero": "trastero" in tl,
                "fecha_publicacion_texto": fecha_publicacion_texto,
                "texto_tarjeta": texto,
                "enlace": enlace,
            })
        except Exception as e:
            print(f"      ⚠ Error parseando tarjeta: {e}")

    # Dedupe dentro del DOM actual.
    unicos = {}
    for r in resultados:
        unicos[clave_anuncio(r)] = r
    return list(unicos.values())


# ============================================================
# FUSIÓN / PERSISTENCIA
# ============================================================

def _valor_valido(v):
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return not (isinstance(v, str) and not v.strip())


def fusionar_registros(registros):
    """Conserva la fila más reciente y rellena sus nulos con datos anteriores."""
    grupos = {}
    for r in registros:
        if not isinstance(r, dict):
            continue
        grupos.setdefault(clave_anuncio(r), []).append(r)

    salida = []
    for _, filas in grupos.items():
        filas.sort(key=lambda x: str(x.get("fecha_extraccion") or ""))
        combinado = dict(filas[-1])
        for vieja in reversed(filas[:-1]):
            for k, v in vieja.items():
                if not _valor_valido(combinado.get(k)) and _valor_valido(v):
                    combinado[k] = v
        fechas = [str(x.get("fecha_extraccion")) for x in filas if _valor_valido(x.get("fecha_extraccion"))]
        if fechas:
            combinado["fecha_primera_extraccion"] = min(fechas)
            combinado["fecha_ultima_extraccion"] = max(fechas)
        salida.append(combinado)
    return salida


def leer_csv_registros(path):
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception as e:
        print(f"⚠ No se pudo leer {path}: {e}")
        return []


def ruta_checkpoint(checkpoint_dir, municipio, zona):
    carpeta = os.path.join(checkpoint_dir, slugify(municipio))
    os.makedirs(carpeta, exist_ok=True)
    return os.path.join(carpeta, f"{slugify(zona or 'todas-las-zonas')}.csv")


def guardar_checkpoint_acumulativo(checkpoint_dir, municipio, zona, nuevos):
    if not nuevos:
        return
    path = ruta_checkpoint(checkpoint_dir, municipio, zona)
    anteriores = leer_csv_registros(path)
    fusionados = fusionar_registros(anteriores + nuevos)
    pd.DataFrame(fusionados).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"      💾 Checkpoint acumulativo: {len(fusionados):,} -> {path}")


def leer_todos_checkpoints(checkpoint_dir):
    registros = []
    if not os.path.exists(checkpoint_dir):
        return registros
    for raiz, _, archivos in os.walk(checkpoint_dir):
        for archivo in archivos:
            if archivo.lower().endswith(".csv"):
                registros.extend(leer_csv_registros(os.path.join(raiz, archivo)))
    return registros


def generar_maestro(rutas):
    # El maestro se reconstruye siempre a partir de todos los checkpoints,
    # de modo que una ejecución interrumpida no pierde lo ya capturado.
    registros = leer_todos_checkpoints(rutas["checkpoint_dir"])

    fusionados = fusionar_registros(registros)
    df = pd.DataFrame(fusionados)
    if not df.empty:
        orden = [c for c in ["municipio", "zona", "precio_eur"] if c in df.columns]
        if orden:
            df = df.sort_values(orden, na_position="last")
        df.to_csv(rutas["output_file"], index=False, encoding="utf-8-sig")
    return df


# ============================================================
# SCROLL ROBUSTO
# ============================================================

def scroll_y_extraer(provincia, municipio, zona, checkpoint_dir):
    acumulados = {}
    rondas_sin_nuevos = 0
    altura_anterior = 0
    rondas_altura_igual = 0

    # Empezamos arriba para no depender de la posición previa del navegador.
    try:
        driver.execute_script("window.scrollTo(0, 0);")
    except Exception:
        pass

    for i in range(1, MAX_SCROLLS + 1):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        actuales = extraer_anuncios(soup, provincia, municipio, zona)
        antes = len(acumulados)
        for r in actuales:
            acumulados[clave_anuncio(r)] = r
        nuevos = len(acumulados) - antes

        if nuevos:
            rondas_sin_nuevos = 0
            guardar_checkpoint_acumulativo(
                checkpoint_dir, municipio, zona, list(acumulados.values())
            )
        else:
            rondas_sin_nuevos += 1

        try:
            y = int(driver.execute_script("return window.pageYOffset || document.documentElement.scrollTop || 0;"))
            alto_vista = int(driver.execute_script("return window.innerHeight || 800;"))
            altura = int(driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"))
        except Exception:
            y, alto_vista, altura = 0, 800, 0

        if altura == altura_anterior:
            rondas_altura_igual += 1
        else:
            rondas_altura_igual = 0
        altura_anterior = altura

        en_final = altura > 0 and (y + alto_vista >= altura - 80)

        print(
            f"      Scroll {i:03d} | DOM {len(actuales):4d} | "
            f"únicos {len(acumulados):5d} | nuevos {nuevos:3d} | "
            f"sin nuevos {rondas_sin_nuevos}/{MAX_RONDAS_SIN_NUEVOS}"
        )

        if en_final and rondas_sin_nuevos >= MAX_RONDAS_SIN_NUEVOS and rondas_altura_igual >= 4:
            break

        try:
            driver.execute_script(
                "window.scrollBy(0, Math.max(500, Math.floor(window.innerHeight * arguments[0])));",
                SCROLL_STEP_RATIO,
            )
        except Exception:
            break

        time.sleep(random.uniform(SCROLL_WAIT_MIN, SCROLL_WAIT_MAX))

        # Al llegar al final, un pequeño empujón ayuda a activar la siguiente carga lazy.
        if en_final:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            except Exception:
                pass
            time.sleep(random.uniform(1.5, 2.5))

    return list(acumulados.values())


# ============================================================
# NAVEGACIÓN / DESCUBRIMIENTO
# ============================================================

def obtener_siguiente_pagina():
    selectores = [
        "a[rel='next']",
        "a[aria-label='Siguiente página']",
        "a[aria-label='Página siguiente']",
        "a[data-testid='next-page']",
    ]
    for selector in selectores:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, selector):
                href = el.get_attribute("href")
                if href:
                    # IMPORTANTE: no quitar ?page=..., etc.
                    return normalizar_url_navegacion(href)
        except Exception:
            pass
    return None


def limpiar_nombre_municipio(texto):
    texto = normalizar_espacios(texto)
    if not texto:
        return None
    patrones = [
        r"(?i)venta de casas y pisos en",
        r"(?i)casas y pisos en venta en",
        r"(?i)casas y pisos de compra en",
        r"(?i)casas y pisos para comprar en",
        r"(?i)comprar casas y pisos en",
        r"(?i)pisos y apartamentos en venta en",
        r"(?i)pisos y apartamentos para comprar en",
        r"(?i)pisos y casas en venta en",
        r"(?i)viviendas en venta en",
        r"(?i)comprar viviendas en",
    ]
    for p in patrones:
        texto = re.sub(p, "", texto)
    texto = texto.strip(" ,:-")
    return texto if 2 <= len(texto) <= 100 else None


def descubrir_municipios_frescos(provincia, url_provincia):
    if not safe_get(url_provincia):
        return []
    time.sleep(random.uniform(2, 3.5))
    aceptar_cookies()

    # Intentamos desplegar listas completas sin seguir enlaces de navegación.
    # Solo pulsamos BUTTONS, nunca <a>, para no abandonar la página de provincia.
    try:
        botones = driver.find_elements(
            By.XPATH,
            "//button[contains(translate(normalize-space(.), "
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ', "
            "'abcdefghijklmnopqrstuvwxyzáéíóú'), 'ver todos')]"
        )
        for boton in botones:
            try:
                driver.execute_script("arguments[0].click();", boton)
                time.sleep(0.35)
            except Exception:
                pass
    except Exception:
        pass

    # Bajamos progresivamente para que se rendericen también bloques del footer.
    for _ in range(12):
        try:
            driver.execute_script("window.scrollBy(0, Math.max(700, window.innerHeight));")
        except Exception:
            break
        time.sleep(0.35)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    encontrados = {}
    for a in soup.find_all("a", href=True):
        href = normalizar_url_navegacion(a.get("href"))
        if not href:
            continue
        path = urlparse(href).path.lower()
        if "/es/comprar/viviendas/" not in path:
            continue
        if "/es/comprar/vivienda/" in path or "/obra-nueva/" in path:
            continue
        if "-provincia/" in path:
            continue
        if not any(x in path for x in ["/todas-las-zonas/l", "/all-zones/l", "/totes-les-zones/l"]):
            continue
        nombre = limpiar_nombre_municipio(a.get_text(" ", strip=True))
        if nombre:
            encontrados[href] = nombre

    return sorted(
        [{"nombre": n, "url": u} for u, n in encontrados.items()],
        key=lambda x: x["nombre"].lower(),
    )


def descubrir_municipios(provincia, url_provincia, estado, estado_file):
    guardados = estado.get("municipios_descubiertos", []) or []
    frescos = []
    if REFRESCAR_MUNICIPIOS or not guardados:
        frescos = descubrir_municipios_frescos(provincia, url_provincia)

    por_url = {}
    for x in guardados + frescos:
        if x.get("url") and x.get("nombre"):
            por_url[normalizar_url_navegacion(x["url"])] = x["nombre"]

    municipios = sorted(
        [{"url": u, "nombre": n} for u, n in por_url.items()],
        key=lambda x: x["nombre"].lower(),
    )
    estado["municipios_descubiertos"] = municipios
    guardar_estado(estado_file, estado)
    print(f"\n✓ Municipios acumulados detectados: {len(municipios)}")
    return municipios


def _enlaces_busqueda_geografica(soup, municipio_slug, url_actual):
    """Devuelve candidatos geográficos del mismo municipio.

    Fotocasa usa rutas como:
      /viviendas/sevilla-capital/todas-las-zonas/l
      /viviendas/sevilla-capital/cerro-amate/l
      /viviendas/sevilla-capital/los-pajaros/l
    Distrito y barrio pueden estar al mismo nivel de URL, por lo que después
    filtramos por bloques/encabezados y no asumimos jerarquía por path.
    """
    actual = normalizar_url_navegacion(url_actual)
    candidatos = {}

    # Primero buscamos secciones del tipo "... en <zona>" y usamos sus enlaces.
    headings = soup.find_all(["h2", "h3"])
    contenedores = []
    for h in headings:
        txt = normalizar_espacios(h.get_text(" ", strip=True)).lower()
        if not txt or "cerca de" in txt or "provincia" in txt or "profundiza" in txt:
            continue
        if any(k in txt for k in ["pisos", "casas", "viviendas", "inmuebles"]):
            cont = h.parent
            for _ in range(3):
                if cont is None:
                    break
                anchors = cont.find_all("a", href=True)
                validos = 0
                for a in anchors:
                    href = normalizar_url_navegacion(a.get("href"))
                    if not href:
                        continue
                    path = urlparse(href).path
                    partes = [p for p in path.split("/") if p]
                    try:
                        i = partes.index("viviendas")
                    except ValueError:
                        continue
                    post = partes[i + 1:]
                    if len(post) == 3 and post[0] == municipio_slug and post[-1] == "l":
                        validos += 1
                if validos >= 2:
                    contenedores.append(cont)
                    break
                cont = cont.parent

    # Si no conseguimos un bloque claro, usamos el documento entero y luego
    # aplicamos filtros conservadores.
    fuentes = contenedores if contenedores else [soup]

    for fuente in fuentes:
        for a in fuente.find_all("a", href=True):
            href = normalizar_url_navegacion(a.get("href"))
            if not href or href == actual:
                continue
            path = urlparse(href).path
            partes = [p for p in path.split("/") if p]
            try:
                i = partes.index("viviendas")
            except ValueError:
                continue
            post = partes[i + 1:]
            if len(post) != 3 or post[0] != municipio_slug or post[-1] != "l":
                continue
            zona_slug = post[1]
            if zona_slug in {"todas-las-zonas", "all-zones", "totes-les-zones", "alle-zonen"}:
                continue
            nombre = limpiar_nombre_municipio(a.get_text(" ", strip=True))
            if not nombre:
                nombre = zona_slug.replace("-", " ").title()
            candidatos[href] = nombre

    return [{"nombre": n, "url": u} for u, n in candidatos.items()]


def descubrir_subzonas(municipio, url_actual):
    if not safe_get(url_actual):
        return [], None
    time.sleep(random.uniform(1.8, 3.0))
    if not esperar_anuncios():
        return [], None

    soup = BeautifulSoup(driver.page_source, "html.parser")
    total = obtener_total_resultados_desde_soup(soup)
    if total is not None:
        print(f"      Fotocasa indica ~{total:,} resultados.")
    if total is not None and total < UMBRAL_DIVIDIR_ZONAS:
        return [], total

    path = urlparse(url_actual).path
    partes = [p for p in path.split("/") if p]
    try:
        i = partes.index("viviendas")
        municipio_slug = partes[i + 1]
    except Exception:
        return [], total

    hijos = _enlaces_busqueda_geografica(soup, municipio_slug, url_actual)

    # Evitamos explosión por enlaces de "cerca de"/siblings: conservamos nombres
    # únicos y como máximo 60 candidatos razonables.
    vistos_nombre = set()
    limpios = []
    for x in sorted(hijos, key=lambda z: z["nombre"].lower()):
        n = slugify(x["nombre"])
        if n in vistos_nombre:
            continue
        vistos_nombre.add(n)
        limpios.append(x)
        if len(limpios) >= 60:
            break

    if len(limpios) >= 2:
        print(f"      ✓ {len(limpios)} zonas/subzonas candidatas.")
        return limpios, total
    return [], total


# ============================================================
# PROCESAMIENTO DE UNA BÚSQUEDA
# ============================================================

def procesar_busqueda_simple(provincia, municipio, zona, url, checkpoint_dir):
    if not safe_get(url):
        raise RuntimeError(f"No se pudo abrir {url}")
    time.sleep(random.uniform(1.8, 3.0))
    if not esperar_anuncios():
        print("      ⚠ No se detectaron tarjetas de anuncios.")
        return [], None

    total_declarado = obtener_total_resultados()
    if total_declarado is not None:
        print(f"      Total declarado por Fotocasa: {total_declarado:,}")

    acumulados = {}
    paginas_visitadas = set()

    for pagina in range(1, MAX_PAGINAS_POR_BUSQUEDA + 1):
        # CONSERVAMOS query params para distinguir páginas reales.
        url_real = normalizar_url_navegacion(driver.current_url)
        if url_real in paginas_visitadas:
            print("      ⚠ URL de página repetida; fin de paginación.")
            break
        paginas_visitadas.add(url_real)

        print(f"      📄 Bloque/página {pagina}: {url_real}")
        datos = scroll_y_extraer(provincia, municipio, zona, checkpoint_dir)
        for r in datos:
            acumulados[clave_anuncio(r)] = r

        siguiente = obtener_siguiente_pagina()
        if not siguiente or siguiente in paginas_visitadas:
            break
        if not safe_get(siguiente):
            break
        time.sleep(random.uniform(PAUSA_BUSQUEDA_MIN, PAUSA_BUSQUEDA_MAX))

    datos = list(acumulados.values())
    guardar_checkpoint_acumulativo(checkpoint_dir, municipio, zona, datos)

    if total_declarado:
        cobertura = 100.0 * len(datos) / total_declarado
        print(
            f"      📈 Vistos en esta búsqueda: {len(datos):,}/{total_declarado:,} "
            f"({cobertura:.1f}% del total declarado; puede haber duplicados/topes)."
        )
    else:
        print(f"      📈 Vistos en esta búsqueda: {len(datos):,}")

    return datos, total_declarado


def procesar_geografia(
    provincia,
    municipio,
    zona,
    url,
    checkpoint_dir,
    errores_file,
    nivel=0,
    visitadas=None,
):
    """Subdivide búsquedas grandes antes de extraerlas."""
    if visitadas is None:
        visitadas = set()
    clave_url = normalizar_url_navegacion(url)
    if clave_url in visitadas:
        return []
    visitadas.add(clave_url)

    print("\n" + "  " * nivel + f"🗺 {municipio} / {zona}")

    try:
        hijos, total = descubrir_subzonas(municipio, url)
    except Exception as e:
        registrar_error(errores_file, provincia, municipio, zona, url, e)
        hijos, total = [], None

    if hijos and nivel < MAX_NIVEL_ZONAS:
        print("  " * nivel + f"   ↳ búsqueda grande; se divide en {len(hijos)} zonas.")
        acumulados = {}
        for i, hijo in enumerate(hijos, 1):
            nombre_hijo = hijo["nombre"]
            try:
                datos = procesar_geografia(
                    provincia,
                    municipio,
                    nombre_hijo,
                    hijo["url"],
                    checkpoint_dir,
                    errores_file,
                    nivel=nivel + 1,
                    visitadas=visitadas,
                )
                for r in datos:
                    acumulados[clave_anuncio(r)] = r
            except Exception as e:
                registrar_error(errores_file, provincia, municipio, nombre_hijo, hijo["url"], e)
                print(f"      ❌ {nombre_hijo}: {e}")
                # Si Chrome ha caído, sin reiniciar aquí el resto de zonas del
                # municipio fallarían en cadena con la sesión muerta. Se reinicia
                # y se reintenta UNA vez la misma zona.
                if es_sesion_muerta(e):
                    reiniciar_driver()
                    safe_get("https://www.fotocasa.es/")
                    aceptar_cookies()
                    visitadas.discard(normalizar_url_navegacion(hijo["url"]))
                    try:
                        datos = procesar_geografia(
                            provincia, municipio, nombre_hijo, hijo["url"],
                            checkpoint_dir, errores_file, nivel=nivel + 1, visitadas=visitadas,
                        )
                        for r in datos:
                            acumulados[clave_anuncio(r)] = r
                    except Exception as e2:
                        registrar_error(errores_file, provincia, municipio, nombre_hijo, hijo["url"], e2)
                        print(f"      ❌ {nombre_hijo} (reintento): {e2}")
            if i < len(hijos):
                time.sleep(random.uniform(PAUSA_BUSQUEDA_MIN, PAUSA_BUSQUEDA_MAX))

        if PROCESAR_PADRE_AUNQUE_HAYA_HIJOS:
            try:
                datos_padre, _ = procesar_busqueda_simple(
                    provincia, municipio, zona, url, checkpoint_dir
                )
                for r in datos_padre:
                    acumulados[clave_anuncio(r)] = r
            except Exception as e:
                registrar_error(errores_file, provincia, municipio, zona, url, e)

        return list(acumulados.values())

    datos, _ = procesar_busqueda_simple(
        provincia, municipio, zona, url, checkpoint_dir
    )
    return datos


# ============================================================
# EJECUCIÓN PROVINCIA
# ============================================================

def ejecutar_provincia(provincia, url_provincia):
    global driver
    rutas = obtener_rutas_provincia(provincia)
    os.makedirs(rutas["runs_dir"], exist_ok=True)
    estado = cargar_estado(rutas["estado_file"])

    inicio = datetime.now()
    print("\n" + "#" * 62)
    print(f"🏠 FOTOCASA — {provincia.upper()}")
    print("#" * 62)
    print(f"Inicio: {inicio:%Y-%m-%d %H:%M:%S}")
    print(f"CSV maestro: {rutas['output_file']}")

    df_inicial = generar_maestro(rutas)
    print(f"Registros acumulados antes de esta pasada: {len(df_inicial):,}")
    del df_inicial

    registros_run = {}

    try:
        reiniciar_driver()
        safe_get("https://www.fotocasa.es/")
        aceptar_cookies()

        municipios = descubrir_municipios(
            provincia, url_provincia, estado, rutas["estado_file"]
        )
        if not municipios:
            raise RuntimeError("No se descubrieron municipios.")

        completadas = set(estado.get("busquedas_completadas", []))
        desde_reinicio = 0

        for idx, m in enumerate(municipios, 1):
            municipio = m["nombre"]
            url = m["url"]
            clave = f"{provincia}|{municipio}"

            if not REESCANEAR_COMPLETADAS and clave in completadas:
                print(f"\n✓ {idx}/{len(municipios)} {municipio}: ya completado; se omite.")
                continue

            if driver is None or desde_reinicio >= MUNICIPIOS_POR_DRIVER:
                reiniciar_driver()
                safe_get("https://www.fotocasa.es/")
                aceptar_cookies()
                desde_reinicio = 0

            print("\n" + "=" * 62)
            print(f"🏘 {idx}/{len(municipios)} — {municipio}")
            print("=" * 62)

            try:
                datos = procesar_geografia(
                    provincia=provincia,
                    municipio=municipio,
                    zona="Todas las zonas",
                    url=url,
                    checkpoint_dir=rutas["checkpoint_dir"],
                    errores_file=rutas["errores_file"],
                )
                for r in datos:
                    registros_run[clave_anuncio(r)] = r
                completadas.add(clave)
                estado["busquedas_completadas"] = sorted(completadas)
                guardar_estado(rutas["estado_file"], estado)
                desde_reinicio += 1
            except Exception as e:
                print(f"❌ ERROR {municipio}: {e}")
                registrar_error(rutas["errores_file"], provincia, municipio, None, url, e)
                cerrar_driver()
                desde_reinicio = MUNICIPIOS_POR_DRIVER

            df_actual = generar_maestro(rutas)
            print(f"📊 Maestro acumulado {provincia}: {len(df_actual):,} anuncios únicos")
            del df_actual
            gc.collect()

            if idx < len(municipios):
                time.sleep(random.uniform(PAUSA_MUNICIPIO_MIN, PAUSA_MUNICIPIO_MAX))

    finally:
        cerrar_driver()

    # Snapshot de ESTA ejecución: nunca se pisa.
    if registros_run:
        stamp = inicio.strftime("%Y%m%d_%H%M%S")
        run_file = os.path.join(rutas["runs_dir"], f"fotocasa_{slugify(provincia)}_{stamp}.csv")
        pd.DataFrame(list(registros_run.values())).to_csv(
            run_file, index=False, encoding="utf-8-sig"
        )
        print(f"\n🧾 Snapshot de la ejecución: {run_file}")

    df_final = generar_maestro(rutas)
    fin = datetime.now()

    print("\n" + "#" * 62)
    print(f"📊 RESULTADO FINAL — {provincia.upper()}")
    print("#" * 62)
    print(f"Anuncios únicos acumulados: {len(df_final):,}")
    print(f"CSV maestro: {rutas['output_file']}")
    print(f"Duración: {fin - inicio}")

    if not df_final.empty and "municipio" in df_final.columns:
        print("\nTop municipios por registros acumulados:")
        print(df_final.groupby("municipio").size().sort_values(ascending=False).head(25).to_string())

    return df_final
