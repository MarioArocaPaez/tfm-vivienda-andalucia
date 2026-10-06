"""
Scraper de detalle de Fotocasa para el dataset de Andalucía.

El script parte de los CSV provinciales generados por los scrapers de listado,
visita cada URL individual y enriquece cada anuncio con información adicional.

Controles principales:
- Detecta anuncios retirados y evita extraer datos de recomendaciones.
- Valida que la ficha abierta corresponde al anuncio esperado.
- Extrae descripción, estado, antigüedad, disponibilidad, certificado
  energético, extras, coordenadas, referencia, agencia, orientación y otros
  campos de detalle.
- Guarda checkpoints frecuentes en el CSV de salida.
- Puede reanudar una ejecución interrumpida.
- Reinicia Chrome de forma preventiva y también tras errores graves del driver.

El código prioriza no introducir datos incorrectos: cuando una ficha no se
puede validar, no se enriquece con información potencialmente ajena.
"""

import time
import random
import re
import os
import json
import html as html_lib
from difflib import SequenceMatcher

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURACIÓN
# ============================================================

SCRAPPERS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRAPPERS_DIR
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

PROVINCIAS_DIR = os.path.join(
    DATA_DIR,
    "provincias"
)

# CSV provinciales generados por los scrapers de listado.
PROVINCIAS = {
    "almeria": "Almería",
    "cadiz": "Cádiz",
    "cordoba": "Córdoba",
    "granada": "Granada",
    "huelva": "Huelva",
    "jaen": "Jaén",
    "malaga": "Málaga",
    "sevilla": "Sevilla",
}

# Único CSV enriquecido de toda Andalucía.
OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "fotocasa_andalucia_detalle.csv"
)

HEADLESS = False

PAUSA_MIN = 3.0
PAUSA_MAX = 6.0

# Persistimos resultados con frecuencia para no perder trabajo.
CHECKPOINT_CADA = 20

# Si el CSV de salida ya existe, solo se procesan enlaces pendientes.
REANUDAR = True

# Procesar TODO el conjunto provincial.
FILA_INICIO = 0
FILA_FIN = None

# Si True, una ficha que no supera la validación de identidad no se
# enriquece. Es la opción conservadora para el TFM.
EXIGIR_VALIDACION = True

# Chrome ha demostrado degradarse en ejecuciones largas. Lo reciclamos
# preventivamente y también cuando detectemos una caída real del driver.
REINICIAR_DRIVER_CADA = 50
MAX_REINTENTOS_ANUNCIO = 3
PAUSA_REINICIO_MIN = 5.0
PAUSA_REINICIO_MAX = 9.0


# ═══════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════

def es_error_driver_grave(error):
    """
    Detecta errores que indican que Chrome/ChromeDriver ha muerto o que
    la sesión ya no se puede reutilizar.
    """

    texto = str(error).lower()

    patrones = [
        "connection aborted",
        "connectionreseterror",
        "connection reset",
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "winerror 10061",
        "winerror 10054",
        "target frame detached",
        "inspector.detached",
        "invalid session id",
        "chrome not reachable",
        "disconnected",
        "session deleted",
        "not connected to devtools",
        "cannot determine loading status",
        "failed to create chrome process",
        "no such window",
    ]

    return any(
        patron in texto
        for patron in patrones
    )


def crear_driver():
    """
    Crea una sesión Chrome limpia.

    Dejamos la configuración sencilla.
    Selenium actual puede gestionar automáticamente ChromeDriver.
    """

    options = webdriver.ChromeOptions()

    if HEADLESS:
        options.add_argument(
            "--headless=new"
        )

    options.add_argument(
        "--window-size=1400,900"
    )

    options.add_argument(
        "--disable-notifications"
    )

    options.add_argument(
        "--disable-extensions"
    )

    options.add_argument(
        "--disable-background-networking"
    )

    options.add_argument(
        "--disable-backgrounding-occluded-windows"
    )

    options.add_argument(
        "--disable-renderer-backgrounding"
    )

    options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
        }
    )

    options.add_argument(
        "--disable-blink-features=AutomationControlled"
    )

    options.add_experimental_option(
        "excludeSwitches",
        ["enable-automation"]
    )

    options.add_experimental_option(
        "useAutomationExtension",
        False
    )

    # No necesitamos fingir un Chrome antiguo.
    # Usamos el User-Agent real del navegador.
    driver = webdriver.Chrome(
        options=options
    )

    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source":
                    "Object.defineProperty("
                    "navigator, 'webdriver', "
                    "{get: () => undefined})"
            }
        )
    except Exception:
        pass

    driver.set_page_load_timeout(
        60
    )

    return driver


def aceptar_cookies(driver):
    try:
        btn = WebDriverWait(
            driver,
            8
        ).until(
            EC.element_to_be_clickable(
                (
                    By.ID,
                    "didomi-notice-agree-button"
                )
            )
        )

        btn.click()

        print(
            "Cookies aceptadas"
        )

        time.sleep(2)

    except Exception:
        print(
            "AVISO: Banner de cookies no encontrado"
        )


# ═══════════════════════════════════════════════════════════════
# HELPERS GENERALES
# ═══════════════════════════════════════════════════════════════

def limpiar_texto(texto):
    if not texto:
        return None

    return " ".join(
        str(texto).split()
    ).strip()


def normalizar_texto(texto):
    """
    Normalización sencilla para comparar títulos/textos.
    """

    if not texto:
        return ""

    texto = html_lib.unescape(
        str(texto)
    ).lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        texto = texto.replace(
            origen,
            destino
        )

    texto = re.sub(
        r"[^a-z0-9]+",
        " ",
        texto
    )

    return " ".join(
        texto.split()
    )


def valor_numero(valor):
    """
    Convierte números procedentes de pandas en float de forma segura.
    """

    if valor is None:
        return None

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    try:
        return float(valor)
    except Exception:
        return None


def ids_equivalentes(a, b):
    """
    Compara IDs aunque pandas haya leído uno como 123.0.
    """

    if a is None or b is None:
        return False

    def canonico(x):
        s = str(x).strip()

        if s.endswith(".0"):
            s = s[:-2]

        return s

    return canonico(a) == canonico(b)


def extraer_id_desde_url(url):
    """
    Ejemplo:
      .../190172547/d
    -> 190172547
    """

    if not url:
        return None

    m = re.search(
        r"/(\d+)/d(?:[/?#]|$)",
        str(url)
    )

    if m:
        return m.group(1)

    # Fallback más laxo.
    m = re.search(
        r"/(\d+)(?:/|$)",
        str(url)
    )

    return (
        m.group(1)
        if m
        else None
    )


def extraer_rango_antiguedad(texto):
    if not texto:
        return None, None

    t = str(
        texto
    ).lower().strip()

    if (
        "obra nueva" in t
        or
        "nuevo" in t
    ):
        return 0, 0

    m = re.search(
        r"(\d+)\s*(?:a|y)\s*(\d+)",
        t
    )

    if m:
        return (
            int(m.group(1)),
            int(m.group(2))
        )

    m = re.search(
        r"m[aá]s\s+de\s+(\d+)",
        t
    )

    if m:
        return (
            int(m.group(1)),
            None
        )

    m = re.search(
        r"menos\s+de\s+(\d+)",
        t
    )

    if m:
        return (
            None,
            int(m.group(1))
        )

    m = re.search(
        r"(\d+)",
        t
    )

    if m:
        v = int(
            m.group(1)
        )

        return v, v

    return None, None


def buscar_valor_tras_etiqueta(
    soup,
    etiqueta_re
):
    """
    Busca el valor HTML que aparece justo después de una etiqueta.

    Se conserva la filosofía del , pero la función se usa sobre
    un bloque de contenido lo más acotado posible.
    """

    for tag in soup.find_all(
        string=re.compile(
            etiqueta_re,
            re.IGNORECASE
        )
    ):
        texto_tag = (
            str(tag).strip()
            if tag
            else ""
        )

        if not texto_tag:
            continue

        parent = tag.parent

        if parent is None:
            continue

        siguiente = (
            parent.find_next_sibling()
        )

        if siguiente:
            val = limpiar_texto(
                siguiente.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                val
                and
                len(val) < 100
                and
                val.lower()
                != texto_tag.lower()
            ):
                return val

        contenedor = parent.parent

        if contenedor:
            texto_cont = limpiar_texto(
                contenedor.get_text(
                    " ",
                    strip=True
                )
            )

            if texto_cont:
                val = (
                    texto_cont
                    .replace(
                        texto_tag,
                        ""
                    )
                    .strip()
                )

                if (
                    val
                    and
                    len(val) < 100
                ):
                    return val

    return None


# ═══════════════════════════════════════════════════════════════
# BLOQUE PRINCIPAL DEL ANUNCIO
# ═══════════════════════════════════════════════════════════════

def obtener_bloque_principal(soup):
    """
    Devuelve el bloque principal donde vive el anuncio.

    Prioridad:
      1. <main>
      2. primer <article> grande
      3. <body>
      4. soup completo

    No garantiza que Fotocasa no incluya recomendaciones dentro,
    pero reduce bastante el universo respecto al HTML completo.
    """

    main = soup.find(
        "main"
    )

    if main:
        return main

    articulos = soup.find_all(
        "article"
    )

    if articulos:
        articulos_ordenados = sorted(
            articulos,
            key=lambda x:
                len(
                    limpiar_texto(
                        x.get_text(
                            " ",
                            strip=True
                        )
                    )
                    or ""
                ),
            reverse=True
        )

        if articulos_ordenados:
            return articulos_ordenados[
                0
            ]

    body = soup.find(
        "body"
    )

    return (
        body
        if body
        else soup
    )


def obtener_texto_cabecera(
    bloque_principal
):
    """
    Intenta quedarse con una zona cercana al H1.

    Esto sirve para validar precio, superficie y habitaciones
    sin mirar toda la página, donde puede haber recomendados.
    """

    h1 = bloque_principal.find(
        "h1"
    )

    if not h1:
        return None, None

    titulo = limpiar_texto(
        h1.get_text(
            " ",
            strip=True
        )
    )

    candidato = h1.parent

    mejor_texto = titulo

    # Subimos pocos niveles.
    # Nos quedamos con un contenedor razonablemente pequeño.
    for _ in range(4):
        if candidato is None:
            break

        texto = limpiar_texto(
            candidato.get_text(
                " ",
                strip=True
            )
        )

        if (
            texto
            and
            len(texto) <= 4500
        ):
            mejor_texto = texto

        candidato = candidato.parent

    return (
        titulo,
        mejor_texto
    )


# ═══════════════════════════════════════════════════════════════
# DETECCIÓN DE ANUNCIO NO DISPONIBLE
# ═══════════════════════════════════════════════════════════════

def detectar_anuncio_no_disponible(
    soup
):
    """
    Detecta la página especial de Fotocasa que aparece cuando
    el anuncio original ha sido retirado.

    Ejemplo real:
        "Ups! Este anuncio ya no está disponible en fotocasa,
         pero te mostramos un listado de anuncios similares"

    En ese caso NO debemos seguir extrayendo datos porque la página
    contiene inmuebles recomendados que NO pertenecen al anuncio
    original.
    """

    try:
        texto_pagina = limpiar_texto(
            soup.get_text(
                " ",
                strip=True
            )
        ) or ""

    except Exception:
        texto_pagina = ""

    texto_lower = texto_pagina.lower()

    patrones = [
        "este anuncio ya no está disponible",
        "este anuncio ya no esta disponible",
        "anuncio ya no está disponible",
        "anuncio ya no esta disponible",
        "te mostramos un listado de anuncios similares",
        "te mostramos anuncios similares",
        "este inmueble ya no está disponible",
        "este inmueble ya no esta disponible",
    ]

    for patron in patrones:
        if patron in texto_lower:
            return (
                True,
                "Fotocasa indica que el anuncio ya no está disponible"
            )

    return (
        False,
        None
    )


# ═══════════════════════════════════════════════════════════════
# VALIDACIÓN DE IDENTIDAD DE LA FICHA
# ═══════════════════════════════════════════════════════════════

def _numero_aparece(
    texto,
    valor,
    tolerancia=0.0
):
    """
    Comprueba si un número esperado aparece en un texto.

    tolerancia=0:
      coincidencia exacta de entero aproximado.

    Para precio permitimos formatos:
      375000
      375.000
      375 000
    """

    valor = valor_numero(
        valor
    )

    if (
        valor is None
        or
        not texto
    ):
        return None

    entero = int(
        round(valor)
    )

    formatos = {
        str(entero),
        f"{entero:,}".replace(
            ",",
            "."
        ),
        f"{entero:,}".replace(
            ",",
            " "
        ),
    }

    texto_normal = str(
        texto
    )

    if any(
        formato in texto_normal
        for formato in formatos
    ):
        return True

    if tolerancia > 0:
        numeros = re.findall(
            r"\b\d{1,7}(?:[.,]\d+)?\b",
            texto_normal
        )

        for numero in numeros:
            try:
                n = float(
                    numero
                    .replace(
                        ".",
                        ""
                    )
                    .replace(
                        ",",
                        "."
                    )
                )
            except ValueError:
                continue

            if abs(
                n - valor
            ) <= tolerancia:
                return True

    return False


def validar_ficha(
    driver,
    soup,
    id_esperado=None,
    titulo_esperado=None,
    precio_esperado=None,
    superficie_esperada=None,
    habitaciones_esperadas=None
):
    """
    Valida que la página cargada corresponde al anuncio esperado.

    Filosofía:
    ---------
    No intentamos demostrar matemáticamente que todo sea idéntico.
    Buscamos evidencias independientes y conservadoras.

    Reglas:
    -------
    - El ID esperado debe coincidir con el ID de la URL real.
    - Además debe coincidir al menos UNA señal de cabecera:
        * título parecido
        * precio
        * superficie
        * habitaciones

    Si no tenemos suficientes datos de entrada para verificar
    cabecera, el ID correcto sigue siendo una evidencia fuerte,
    pero la ficha se marca como validada solo si existe H1.
    """

    url_real = driver.current_url

    id_url = extraer_id_desde_url(
        url_real
    )

    bloque_principal = obtener_bloque_principal(
        soup
    )

    titulo_detalle, texto_cabecera = (
        obtener_texto_cabecera(
            bloque_principal
        )
    )

    razones = []
    puntos = 0
    senales_disponibles = 0
    senales_coincidentes = 0

    # --------------------------------------------------------
    # 1. ID
    # --------------------------------------------------------

    if id_esperado is not None:
        if not ids_equivalentes(
            id_esperado,
            id_url
        ):
            return {
                "validado":
                    False,

                "motivo":
                    (
                        f"ID distinto: "
                        f"esperado={id_esperado}, "
                        f"url={id_url}"
                    ),

                "titulo_detalle":
                    titulo_detalle,

                "texto_cabecera":
                    texto_cabecera,
            }

        puntos += 3

        razones.append(
            "ID correcto"
        )

    # --------------------------------------------------------
    # 2. TÍTULO
    # --------------------------------------------------------

    if (
        titulo_esperado
        and
        titulo_detalle
    ):
        senales_disponibles += 1

        a = normalizar_texto(
            titulo_esperado
        )

        b = normalizar_texto(
            titulo_detalle
        )

        ratio = SequenceMatcher(
            None,
            a,
            b
        ).ratio()

        # También aceptamos coincidencia parcial razonable.
        tokens_a = set(
            a.split()
        )

        tokens_b = set(
            b.split()
        )

        union = (
            tokens_a
            |
            tokens_b
        )

        inter = (
            tokens_a
            &
            tokens_b
        )

        jaccard = (
            len(inter)
            /
            len(union)
            if union
            else 0
        )

        if (
            ratio >= 0.55
            or
            jaccard >= 0.35
        ):
            senales_coincidentes += 1
            puntos += 1

            razones.append(
                "título compatible"
            )

    # --------------------------------------------------------
    # 3. PRECIO
    # --------------------------------------------------------

    if (
        precio_esperado is not None
        and
        not pd.isna(
            precio_esperado
        )
    ):
        senales_disponibles += 1

        if _numero_aparece(
            texto_cabecera,
            precio_esperado
        ):
            senales_coincidentes += 1
            puntos += 1

            razones.append(
                "precio compatible"
            )

    # --------------------------------------------------------
    # 4. SUPERFICIE
    # --------------------------------------------------------

    if (
        superficie_esperada is not None
        and
        not pd.isna(
            superficie_esperada
        )
    ):
        senales_disponibles += 1

        superficie = int(
            round(
                float(
                    superficie_esperada
                )
            )
        )

        patron = re.compile(
            rf"\b{superficie}\s*m[²2]\b",
            re.IGNORECASE
        )

        if (
            texto_cabecera
            and
            patron.search(
                texto_cabecera
            )
        ):
            senales_coincidentes += 1
            puntos += 1

            razones.append(
                "superficie compatible"
            )

    # --------------------------------------------------------
    # 5. HABITACIONES
    # --------------------------------------------------------

    if (
        habitaciones_esperadas is not None
        and
        not pd.isna(
            habitaciones_esperadas
        )
    ):
        senales_disponibles += 1

        habs = int(
            round(
                float(
                    habitaciones_esperadas
                )
            )
        )

        patron = re.compile(
            rf"\b{habs}\s*hab",
            re.IGNORECASE
        )

        if (
            texto_cabecera
            and
            patron.search(
                texto_cabecera
            )
        ):
            senales_coincidentes += 1
            puntos += 1

            razones.append(
                "habitaciones compatibles"
            )

    # --------------------------------------------------------
    # DECISIÓN
    # --------------------------------------------------------

    if not titulo_detalle:
        return {
            "validado":
                False,

            "motivo":
                (
                    "ID correcto pero no se encontró "
                    "H1/título principal"
                ),

            "titulo_detalle":
                titulo_detalle,

            "texto_cabecera":
                texto_cabecera,
        }

    # Si teníamos señales que comparar, exigimos al menos una.
    if (
        senales_disponibles > 0
        and
        senales_coincidentes == 0
    ):
        return {
            "validado":
                False,

            "motivo":
                (
                    "ID correcto pero ninguna señal "
                    "de cabecera coincide "
                    f"({senales_disponibles} comprobadas)"
                ),

            "titulo_detalle":
                titulo_detalle,

            "texto_cabecera":
                texto_cabecera,
        }

    return {
        "validado":
            True,

        "motivo":
            " | ".join(
                razones
            ),

        "titulo_detalle":
            titulo_detalle,

        "texto_cabecera":
            texto_cabecera,
    }


# ═══════════════════════════════════════════════════════════════
# DESCRIPCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def extraer_descripcion_principal(
    soup
):
    """
    Extrae la descripción del inmueble principal.

     hacía:
        - buscar todos los <p>/<div>
        - quedarse con el texto más largo

    Eso podía seleccionar un anuncio recomendado.

    :
        1. busca contenedores semánticos de descripción
        2. busca encabezados "Descripción", "Más detalles", etc.
        3. limita la búsqueda al bloque principal
        4. aplica filtros para evitar módulos comerciales/recomendados
    """

    bloque = obtener_bloque_principal(
        soup
    )

    # --------------------------------------------------------
    # VÍA 1: atributos semánticos
    # --------------------------------------------------------

    selectores = [
        "[data-testid*='description']",
        "[data-testid*='Description']",
        "[class*='description']",
        "[class*='Description']",
        "[id*='description']",
        "[id*='Description']",
    ]

    candidatos = []

    for selector in selectores:
        try:
            elementos = bloque.select(
                selector
            )
        except Exception:
            elementos = []

        for elemento in elementos:
            texto = limpiar_texto(
                elemento.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                texto
                and
                120 <= len(texto) <= 8000
            ):
                candidatos.append(
                    texto
                )

    # --------------------------------------------------------
    # VÍA 2: encabezados conocidos
    # --------------------------------------------------------

    patron_heading = re.compile(
        r"^("
        r"descripci[oó]n"
        r"|m[aá]s detalles"
        r"|detalle del inmueble"
        r"|descripci[oó]n del inmueble"
        r")$",
        re.IGNORECASE
    )

    for heading in bloque.find_all(
        [
            "h2",
            "h3",
            "h4"
        ]
    ):
        texto_heading = limpiar_texto(
            heading.get_text(
                " ",
                strip=True
            )
        )

        if (
            not texto_heading
            or
            not patron_heading.search(
                texto_heading
            )
        ):
            continue

        # Buscar hermanos posteriores hasta el siguiente heading.
        actual = heading.find_next_sibling()

        partes = []
        pasos = 0

        while (
            actual is not None
            and
            pasos < 15
        ):
            if (
                getattr(
                    actual,
                    "name",
                    None
                )
                in [
                    "h1",
                    "h2",
                    "h3"
                ]
            ):
                break

            texto = limpiar_texto(
                actual.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                texto
                and
                len(texto) >= 60
            ):
                partes.append(
                    texto
                )

            actual = (
                actual.find_next_sibling()
            )

            pasos += 1

        if partes:
            combinado = limpiar_texto(
                " ".join(
                    partes
                )
            )

            if (
                combinado
                and
                120 <= len(combinado) <= 8000
            ):
                candidatos.append(
                    combinado
                )

    # --------------------------------------------------------
    # VÍA 3: "Leer más"
    # --------------------------------------------------------

    for texto in bloque.find_all(
        string=re.compile(
            r"leer más",
            re.IGNORECASE
        )
    ):
        contenedor = texto.parent

        for _ in range(4):
            if contenedor is None:
                break

            contenido = limpiar_texto(
                contenedor.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                contenido
                and
                150 <= len(contenido) <= 8000
            ):
                candidatos.append(
                    contenido
                )

                break

            contenedor = (
                contenedor.parent
            )

    # --------------------------------------------------------
    # FILTROS
    # --------------------------------------------------------

    filtrados = []

    for texto in candidatos:
        t = texto.lower()

        # Evitar contenedores que claramente son agregados
        # de recomendados/carruseles.
        penalizadores = [
            "otros inmuebles",
            "también te puede interesar",
            "tambien te puede interesar",
            "inmuebles similares",
            "anuncios similares",
            "ver más anuncios",
            "ver mas anuncios",
        ]

        if any(
            p in t
            for p in penalizadores
        ):
            continue

        filtrados.append(
            texto
        )

    if not filtrados:
        return None

    # Preferimos el texto más largo ENTRE candidatos ya acotados,
    # no entre todos los DIV de la página.
    return max(
        filtrados,
        key=len
    )


# ═══════════════════════════════════════════════════════════════
# CERTIFICADO ENERGÉTICO
# ═══════════════════════════════════════════════════════════════

def extraer_certificado_energetico(
    html_objetivo
):
    """
    Extrae:
      - letra
      - consumo kWh/m²/año
      - emisiones kg CO₂/m²/año

    La entrada debe ser, cuando sea posible, HTML del bloque principal,
    no el page_source completo.

    999 se considera centinela y se descarta.
    """

    letra = None
    consumo = None
    emision = None

    if not html_objetivo:
        return (
            letra,
            consumo,
            emision
        )

    # --------------------------------------------------------
    # VÍA 1: JSON-LD dentro del bloque recibido
    # --------------------------------------------------------

    for bloque in re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>'
        r'(.*?)</script>',
        html_objetivo,
        re.DOTALL | re.IGNORECASE
    ):
        try:
            datos = json.loads(
                bloque
            )

            objetos = (
                datos
                if isinstance(
                    datos,
                    list
                )
                else [datos]
            )

            for obj in objetos:
                if not isinstance(
                    obj,
                    dict
                ):
                    continue

                rating = obj.get(
                    "energyEfficiencyRating",
                    ""
                )

                if (
                    rating
                    and
                    re.match(
                        r"^[A-G]$",
                        str(rating).strip()
                    )
                ):
                    letra = str(
                        rating
                    ).strip().upper()

                    break

        except Exception:
            pass

        if letra:
            break

    # --------------------------------------------------------
    # Ventana de contexto
    # --------------------------------------------------------

    ventana = ""

    for patron in [
        r"calificaci[oó]n\s+energ[eé]tica.{0,800}",
        r"etiqueta\s+de\s+calificaci[oó]n.{0,800}",
        r"eficiencia\s+energ[eé]tica.{0,800}",
        r"certificado\s+energ[eé]tico.{0,800}",
    ]:
        m = re.search(
            patron,
            html_objetivo,
            re.IGNORECASE | re.DOTALL
        )

        if m:
            ventana += (
                " "
                + m.group(0)
            )

    # --------------------------------------------------------
    # Letra
    # --------------------------------------------------------

    if (
        not letra
        and
        ventana
    ):
        for patron in [
            r"energ[eé]tica[:\s]+([A-G])\b",
            r"\b([A-G])\s*Emisiones",
            r"letra\s+([A-G])\b",
            r"clase\s+energ[eé]tica[:\s]+([A-G])\b",
        ]:
            m = re.search(
                patron,
                ventana,
                re.IGNORECASE
            )

            if m:
                letra = (
                    m.group(1)
                    .upper()
                )

                break

    # --------------------------------------------------------
    # Consumo
    # --------------------------------------------------------

    if ventana:
        m = re.search(
            r"(\d{1,4}(?:[.,]\d+)?)\s*"
            r"kWh\s*/?\s*m[²2]\s*/?\s*a[ñn]o",
            ventana,
            re.IGNORECASE
        )

        if m:
            try:
                v = float(
                    m.group(1)
                    .replace(
                        ".",
                        ""
                    )
                    .replace(
                        ",",
                        "."
                    )
                )

                if (
                    1 <= v <= 2000
                    and
                    v != 999
                ):
                    consumo = v

            except ValueError:
                pass

    # --------------------------------------------------------
    # Emisiones
    # --------------------------------------------------------

    if ventana:
        for patron in [
            r"(\d{1,4}(?:[.,]\d+)?)\s*"
            r"kg\s*CO[2₂]\s*/?\s*m[²2]\s*/?\s*a[ñn]o",

            r"[Ee]misiones[:\s]+"
            r"(\d{1,4}(?:[.,]\d+)?)\s*kg",

            r"\b(\d{1,4}(?:[.,]\d+)?)\s*kg\s*CO",
        ]:
            m = re.search(
                patron,
                ventana,
                re.IGNORECASE
            )

            if not m:
                continue

            try:
                v = float(
                    m.group(1)
                    .replace(
                        ".",
                        ""
                    )
                    .replace(
                        ",",
                        "."
                    )
                )

                # 999 NUNCA se acepta.
                if (
                    1 <= v <= 1000
                    and
                    v != 999
                ):
                    emision = v
                    break

            except ValueError:
                pass

    return (
        letra,
        consumo,
        emision
    )


# ═══════════════════════════════════════════════════════════════
# COORDENADAS GPS
# ═══════════════════════════════════════════════════════════════

def _lat_lon_de_dict(
    obj
):
    """
    Busca latitud/longitud dentro del MISMO diccionario.

    Evita el problema :
      coger una latitud de un objeto
      y una longitud de otro.
    """

    if not isinstance(
        obj,
        dict
    ):
        return (
            None,
            None
        )

    claves_lat = [
        "latitude",
        "lat"
    ]

    claves_lon = [
        "longitude",
        "lng",
        "lon"
    ]

    lat = None
    lon = None

    for clave in claves_lat:
        if clave in obj:
            try:
                v = float(
                    obj[
                        clave
                    ]
                )

                if 35 < v < 45:
                    lat = v
                    break

            except Exception:
                pass

    for clave in claves_lon:
        if clave in obj:
            try:
                v = float(
                    obj[
                        clave
                    ]
                )

                if -10 < v < 5:
                    lon = v
                    break

            except Exception:
                pass

    if (
        lat is not None
        and
        lon is not None
    ):
        return (
            lat,
            lon
        )

    # Algunos JSON usan:
    #   geo: { latitude, longitude }
    # o:
    #   location: { lat, lng }
    for clave in [
        "geo",
        "location",
        "coordinates",
        "coordinate",
        "position",
    ]:
        hijo = obj.get(
            clave
        )

        if isinstance(
            hijo,
            dict
        ):
            lat2, lon2 = (
                _lat_lon_de_dict(
                    hijo
                )
            )

            if (
                lat2 is not None
                and
                lon2 is not None
            ):
                return (
                    lat2,
                    lon2
                )

    return (
        None,
        None
    )


def _objeto_contiene_id(
    obj,
    id_esperado
):
    if id_esperado is None:
        return False

    esperado = str(
        id_esperado
    ).replace(
        ".0",
        ""
    )

    if isinstance(
        obj,
        dict
    ):
        for clave, valor in obj.items():
            clave_l = str(
                clave
            ).lower()

            if (
                "id"
                in clave_l
                or
                "reference"
                in clave_l
                or
                "url"
                in clave_l
            ):
                if esperado in str(
                    valor
                ):
                    return True

    try:
        serializado = json.dumps(
            obj,
            ensure_ascii=False
        )

        return esperado in serializado

    except Exception:
        return False


def _buscar_coords_en_objeto(
    obj,
    id_esperado=None,
    exigir_id=False
):
    """
    Recorre JSON recursivamente.

    Si exigir_id=True, solo acepta coordenadas dentro de un objeto
    que además contenga el ID esperado.
    """

    if isinstance(
        obj,
        dict
    ):
        contiene_id = (
            _objeto_contiene_id(
                obj,
                id_esperado
            )
        )

        lat, lon = (
            _lat_lon_de_dict(
                obj
            )
        )

        if (
            lat is not None
            and
            lon is not None
            and
            (
                not exigir_id
                or
                contiene_id
            )
        ):
            return (
                lat,
                lon
            )

        for valor in obj.values():
            encontrado = (
                _buscar_coords_en_objeto(
                    valor,
                    id_esperado=id_esperado,
                    exigir_id=exigir_id
                )
            )

            if (
                encontrado[0]
                is not None
            ):
                return encontrado

    elif isinstance(
        obj,
        list
    ):
        for elemento in obj:
            encontrado = (
                _buscar_coords_en_objeto(
                    elemento,
                    id_esperado=id_esperado,
                    exigir_id=exigir_id
                )
            )

            if (
                encontrado[0]
                is not None
            ):
                return encontrado

    return (
        None,
        None
    )


def extraer_coordenadas(
    page_source,
    id_esperado=None,
    url_esperada=None
):
    """
    Busca GPS en orden de fiabilidad:

      1. JSON-LD asociado al anuncio
      2. __NEXT_DATA__ objeto que contiene el ID
      3. cualquier JSON estructurado que contenga el ID
      4. JSON-LD genérico SOLO si hay un único par de coordenadas

    Se elimina el fallback  de:
        "primera latitud del source"
        +
        "primera longitud del source"

    porque podía mezclar inmuebles recomendados.
    """

    # --------------------------------------------------------
    # 1. JSON-LD
    # --------------------------------------------------------

    pares_jsonld = []

    for bloque in re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>'
        r'(.*?)</script>',
        page_source,
        re.DOTALL | re.IGNORECASE
    ):
        try:
            datos = json.loads(
                bloque
            )
        except Exception:
            continue

        objetos = (
            datos
            if isinstance(
                datos,
                list
            )
            else [datos]
        )

        for obj in objetos:
            if not isinstance(
                obj,
                dict
            ):
                continue

            contiene_id = (
                _objeto_contiene_id(
                    obj,
                    id_esperado
                )
            )

            if (
                not contiene_id
                and
                url_esperada
            ):
                try:
                    contiene_id = (
                        str(
                            url_esperada
                        )
                        in json.dumps(
                            obj,
                            ensure_ascii=False
                        )
                    )
                except Exception:
                    pass

            lat, lon = (
                _buscar_coords_en_objeto(
                    obj,
                    id_esperado=id_esperado,
                    exigir_id=False
                )
            )

            if (
                lat is not None
                and
                lon is not None
            ):
                if contiene_id:
                    return (
                        lat,
                        lon
                    )

                pares_jsonld.append(
                    (
                        lat,
                        lon
                    )
                )

    # --------------------------------------------------------
    # 2. __NEXT_DATA__ asociado al ID
    # --------------------------------------------------------

    m = re.search(
        r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>'
        r'(.*?)</script>',
        page_source,
        re.DOTALL | re.IGNORECASE
    )

    if m:
        try:
            datos_next = json.loads(
                m.group(1)
            )

            lat, lon = (
                _buscar_coords_en_objeto(
                    datos_next,
                    id_esperado=id_esperado,
                    exigir_id=True
                )
            )

            if (
                lat is not None
                and
                lon is not None
            ):
                return (
                    lat,
                    lon
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # 3. Scripts JSON/JS que contienen el ID
    # --------------------------------------------------------

    esperado = (
        str(id_esperado)
        .replace(
            ".0",
            ""
        )
        if id_esperado is not None
        else None
    )

    if esperado:
        for script in re.findall(
            r"<script[^>]*>(.*?)</script>",
            page_source,
            re.DOTALL | re.IGNORECASE
        ):
            if esperado not in script:
                continue

            # Ventana alrededor del ID.
            posicion = script.find(
                esperado
            )

            inicio = max(
                0,
                posicion - 5000
            )

            fin = min(
                len(script),
                posicion + 5000
            )

            ventana = script[
                inicio:fin
            ]

            # Pares cercanos dentro de la MISMA ventana.
            patrones_pares = [
                (
                    r'"latitude"\s*:\s*"?([-\d.]+)"?'
                    r'.{0,800}?'
                    r'"longitude"\s*:\s*"?([-\d.]+)"?'
                ),
                (
                    r'"lat"\s*:\s*"?([-\d.]+)"?'
                    r'.{0,800}?'
                    r'"lng"\s*:\s*"?([-\d.]+)"?'
                ),
            ]

            for patron in patrones_pares:
                mp = re.search(
                    patron,
                    ventana,
                    re.DOTALL | re.IGNORECASE
                )

                if not mp:
                    continue

                try:
                    lat = float(
                        mp.group(1)
                    )

                    lon = float(
                        mp.group(2)
                    )

                    if (
                        35 < lat < 45
                        and
                        -10 < lon < 5
                    ):
                        return (
                            lat,
                            lon
                        )

                except Exception:
                    pass

    # --------------------------------------------------------
    # 4. JSON-LD genérico SOLO si es inequívoco
    # --------------------------------------------------------

    pares_unicos = list(
        dict.fromkeys(
            pares_jsonld
        )
    )

    if len(
        pares_unicos
    ) == 1:
        return pares_unicos[
            0
        ]

    return (
        None,
        None
    )


# ═══════════════════════════════════════════════════════════════
# EXTRAS
# ═══════════════════════════════════════════════════════════════

EXTRAS_CONOCIDOS = [
    "patio",
    "terraza",
    "balcón",
    "balcon",
    "jardín",
    "jardin",
    "piscina",
    "garaje",
    "parking",
    "trastero",
    "ascensor",
    "aire acondicionado",
    "calefacción",
    "calefaccion",
    "cocina equipada",
    "electrodomésticos",
    "electrodomesticos",
    "armarios empotrados",
    "puerta blindada",
    "domótica",
    "vistas al mar",
    "primera línea",
    "portero físico",
    "videovigilancia",
    "suelo radiante",
    "chimenea",
    "obra nueva",
    "a estrenar",
    "exterior",
    "interior",
    "luminoso",
    "reformado",
    "sin amueblar",
    "amueblado",
]


def extraer_extras(
    bloque_principal
):
    """
    Mantiene la estrategia del  pero trabaja sobre el bloque
    principal en lugar de la página completa.
    """

    extras = set()

    bloques_extras = []

    for heading in bloque_principal.find_all(
        string=re.compile(
            r"equipamiento|características|"
            r"extras|más características",
            re.IGNORECASE
        )
    ):
        padre = heading.parent

        for _ in range(4):
            if padre is None:
                break

            padre = padre.parent

        if padre:
            bloques_extras.append(
                padre
            )

    if bloques_extras:
        candidatos_li = []

        for bloque in bloques_extras:
            candidatos_li.extend(
                bloque.find_all(
                    "li"
                )
            )

        max_len = 80

    else:
        candidatos_li = (
            bloque_principal.find_all(
                "li"
            )
        )

        max_len = 40

    for li in candidatos_li:
        txt = limpiar_texto(
            li.get_text(
                " ",
                strip=True
            )
            or ""
        )

        if not txt:
            continue

        txt_l = txt.lower()

        for extra in EXTRAS_CONOCIDOS:
            if (
                extra in txt_l
                and
                len(txt) < max_len
            ):
                extras.add(
                    extra
                )

    if not extras:
        return None

    return " | ".join(
        sorted(
            extras
        )
    )


# ═══════════════════════════════════════════════════════════════
# AGENCIA / ANUNCIANTE
# ═══════════════════════════════════════════════════════════════

def extraer_agencia(
    bloque_principal
):
    """
    Extracción conservadora del anunciante.

    Objetivo:
      - aceptar nombres claramente asociados al anunciante
      - evitar textos genéricos de navegación o legales
      - evitar marcar "Particular" por encontrar esa palabra
        en cualquier parte de la página

    Si no hay suficiente evidencia, devolvemos None.
    Es mejor tener un valor ausente que uno incorrecto.
    """

    if bloque_principal is None:
        return None

    textos_invalidos = {
        "aviso legal",
        "política de privacidad",
        "politica de privacidad",
        "contactar",
        "llamar",
        "ver teléfono",
        "ver telefono",
        "pedir más datos",
        "pedir mas datos",
        "fotocasa",
        "guardar",
        "compartir",
        "favorito",
        "descartar",
        "más detalles",
        "mas detalles",
        "leer más",
        "leer mas",
        "anunciante",
        "profesional",
    }

    # --------------------------------------------------------
    # VÍA 1: imágenes/logos cuyo ALT parece realmente
    #        un nombre de inmobiliaria/agencia
    # --------------------------------------------------------

    for img in bloque_principal.find_all(
        "img",
        alt=True
    ):
        alt = limpiar_texto(
            img.get(
                "alt",
                ""
            )
        )

        if not alt:
            continue

        alt_l = alt.lower()

        if (
            alt_l in textos_invalidos
            or
            len(alt) < 4
            or
            len(alt) > 100
        ):
            continue

        if any(
            kw in alt_l
            for kw in [
                "inmobiliaria",
                "agencia",
                "real estate",
                "properties",
                "property",
                "gilmar",
                "tecnocasa",
                "redpiso",
                "remax",
                "re/max",
                "engel",
                "volkers",
            ]
        ):
            return alt

    # --------------------------------------------------------
    # VÍA 2: localizar una sección explícita de anunciante
    # --------------------------------------------------------

    etiquetas_anunciante = bloque_principal.find_all(
        string=re.compile(
            r"^(anunciante|profesional|particular)$",
            re.IGNORECASE
        )
    )

    for etiqueta in etiquetas_anunciante:
        contenedor = etiqueta.parent

        # Subimos poco para no tragarnos media página.
        for _ in range(3):
            if contenedor is None:
                break

            texto_contenedor = limpiar_texto(
                contenedor.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                texto_contenedor
                and
                len(texto_contenedor) <= 600
            ):
                texto_lower = texto_contenedor.lower()

                # Particular solo si aparece dentro de este
                # bloque explícitamente asociado al anunciante.
                if re.search(
                    r"\bparticular\b",
                    texto_lower
                ):
                    return "Particular"

                candidatos = contenedor.find_all(
                    [
                        "strong",
                        "span",
                        "a",
                        "p",
                    ]
                )

                for child in candidatos:
                    candidato = limpiar_texto(
                        child.get_text(
                            " ",
                            strip=True
                        )
                    )

                    if not candidato:
                        continue

                    candidato_l = candidato.lower()

                    if (
                        candidato_l in textos_invalidos
                        or
                        len(candidato) < 4
                        or
                        len(candidato) > 100
                    ):
                        continue

                    # Evitar frases claramente de interfaz.
                    if any(
                        trozo in candidato_l
                        for trozo in [
                            "ver teléfono",
                            "ver telefono",
                            "contactar",
                            "pedir más datos",
                            "pedir mas datos",
                            "aviso legal",
                            "política",
                            "politica",
                            "cookies",
                        ]
                    ):
                        continue

                    # Para reducir falsos positivos, exigimos que
                    # el candidato no sea una frase demasiado larga.
                    if len(
                        candidato.split()
                    ) <= 10:
                        return candidato

            contenedor = contenedor.parent

    # --------------------------------------------------------
    # VÍA 3: buscar marcas evidentes en bloques pequeños cercanos
    #        a "Contactar" / "Ver teléfono"
    # --------------------------------------------------------

    for tag in bloque_principal.find_all(
        string=re.compile(
            r"contactar|ver tel[eé]fono",
            re.IGNORECASE
        )
    ):
        contenedor = tag.parent

        for _ in range(3):
            if contenedor is None:
                break

            texto_contenedor = limpiar_texto(
                contenedor.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                texto_contenedor
                and
                len(texto_contenedor) <= 500
            ):
                for child in contenedor.find_all(
                    [
                        "strong",
                        "span",
                        "a",
                    ]
                ):
                    candidato = limpiar_texto(
                        child.get_text(
                            " ",
                            strip=True
                        )
                    )

                    if not candidato:
                        continue

                    candidato_l = candidato.lower()

                    if (
                        candidato_l in textos_invalidos
                        or
                        len(candidato) < 4
                        or
                        len(candidato) > 100
                    ):
                        continue

                    if any(
                        kw in candidato_l
                        for kw in [
                            "inmobiliaria",
                            "agencia",
                            "real estate",
                            "properties",
                            "gilmar",
                            "tecnocasa",
                            "redpiso",
                            "remax",
                            "re/max",
                        ]
                    ):
                        return candidato

            contenedor = contenedor.parent

    return None


# ═══════════════════════════════════════════════════════════════
# EXTRACTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def extraer_detalle(
    driver,
    url,
    id_esperado=None,
    titulo_esperado=None,
    precio_esperado=None,
    superficie_esperada=None,
    habitaciones_esperadas=None
):
    """
    Extrae detalles de UN anuncio y valida primero su identidad.
    """

    resultado = {
        "url_final":
            None,

        "anuncio_disponible":
            None,

        "motivo_no_disponible":
            None,

        "titulo_detalle":
            None,

        "descripcion":
            None,

        "estado":
            None,

        "antiguedad":
            None,

        "antiguedad_min_anos":
            None,

        "antiguedad_max_anos":
            None,

        "amueblado":
            None,

        "disponibilidad":
            None,

        "cert_energia_letra":
            None,

        "cert_energia_consumo":
            None,

        "cert_energia_emision":
            None,

        "extras":
            None,

        "num_fotos":
            None,

        "latitud":
            None,

        "longitud":
            None,

        "referencia":
            None,

        "agencia":
            None,

        "orientacion":
            None,

        "superficie_terreno_m2":
            None,

        "bajada_precio_eur":
            None,

        "tiene_bajada_precio":
            False,

        "detalle_ok":
            False,

        "detalle_validado":
            False,

        "motivo_validacion":
            None,
    }

    try:
        # --------------------------------------------------------
        # CARGAR FICHA
        # --------------------------------------------------------

        driver.get(
            url
        )

        WebDriverWait(
            driver,
            15
        ).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "h1, main"
                )
            )
        )

        time.sleep(
            random.uniform(
                2.5,
                4.0
            )
        )

        page_source = (
            driver.page_source
        )

        soup = BeautifulSoup(
            page_source,
            "html.parser"
        )

        # --------------------------------------------------------
        # URL FINAL
        # --------------------------------------------------------

        resultado[
            "url_final"
        ] = driver.current_url

        # --------------------------------------------------------
        # DETECTAR ANUNCIO RETIRADO / NO DISPONIBLE
        # --------------------------------------------------------
        #
        # Fotocasa puede mantener viva la URL del anuncio pero
        # sustituir la ficha por:
        #
        #   "Este anuncio ya no está disponible..."
        #
        # seguida de un listado de inmuebles similares.
        #
        # Si seguimos scrapeando en ese punto, contaminamos el
        # registro original con datos de otras viviendas.
        # --------------------------------------------------------

        no_disponible, motivo_no_disponible = (
            detectar_anuncio_no_disponible(
                soup
            )
        )

        if no_disponible:
            resultado[
                "anuncio_disponible"
            ] = False

            resultado[
                "motivo_no_disponible"
            ] = motivo_no_disponible

            resultado[
                "detalle_validado"
            ] = False

            resultado[
                "detalle_ok"
            ] = False

            resultado[
                "motivo_validacion"
            ] = motivo_no_disponible

            print(
                "      AVISO: Anuncio retirado de Fotocasa"
            )

            return resultado

        # Si no aparece el mensaje de retirada, tratamos la ficha
        # como potencialmente disponible. La validación posterior
        # decidirá si realmente pertenece al ID esperado.
        resultado[
            "anuncio_disponible"
        ] = True

        bloque_principal = (
            obtener_bloque_principal(
                soup
            )
        )

        # --------------------------------------------------------
        # VALIDAR IDENTIDAD ANTES DE EXTRAER DETALLES
        # --------------------------------------------------------

        validacion = validar_ficha(
            driver=driver,
            soup=soup,
            id_esperado=id_esperado,
            titulo_esperado=titulo_esperado,
            precio_esperado=precio_esperado,
            superficie_esperada=superficie_esperada,
            habitaciones_esperadas=habitaciones_esperadas
        )

        resultado[
            "titulo_detalle"
        ] = validacion[
            "titulo_detalle"
        ]

        resultado[
            "detalle_validado"
        ] = validacion[
            "validado"
        ]

        resultado[
            "motivo_validacion"
        ] = validacion[
            "motivo"
        ]

        if (
            EXIGIR_VALIDACION
            and
            not resultado[
                "detalle_validado"
            ]
        ):
            print(
                "      AVISO: Ficha rechazada: "
                f"{resultado['motivo_validacion']}"
            )

            return resultado

        # --------------------------------------------------------
        # HTML DEL BLOQUE PRINCIPAL
        # --------------------------------------------------------

        html_principal = str(
            bloque_principal
        )

        # --------------------------------------------------------
        # DESCRIPCIÓN
        # --------------------------------------------------------

        resultado[
            "descripcion"
        ] = extraer_descripcion_principal(
            soup
        )

        # --------------------------------------------------------
        # CARACTERÍSTICAS
        # --------------------------------------------------------

        estado_raw = (
            buscar_valor_tras_etiqueta(
                bloque_principal,
                r"^estado$"
            )
        )

        if estado_raw:
            resultado[
                "estado"
            ] = estado_raw

        antig_raw = (
            buscar_valor_tras_etiqueta(
                bloque_principal,
                r"antig[uü]edad"
            )
        )

        if antig_raw:
            resultado[
                "antiguedad"
            ] = antig_raw

            mn, mx = (
                extraer_rango_antiguedad(
                    antig_raw
                )
            )

            resultado[
                "antiguedad_min_anos"
            ] = mn

            resultado[
                "antiguedad_max_anos"
            ] = mx

        amueblado_raw = (
            buscar_valor_tras_etiqueta(
                bloque_principal,
                r"^amueblado$"
            )
        )

        if amueblado_raw:
            resultado[
                "amueblado"
            ] = amueblado_raw

        disp_raw = (
            buscar_valor_tras_etiqueta(
                bloque_principal,
                r"disponibilidad"
            )
        )

        if disp_raw:
            resultado[
                "disponibilidad"
            ] = disp_raw

        orient_raw = (
            buscar_valor_tras_etiqueta(
                bloque_principal,
                r"^orientaci[oó]n$"
            )
        )

        if not orient_raw:
            texto_principal = limpiar_texto(
                bloque_principal.get_text(
                    " ",
                    strip=True
                )
            ) or ""

            m_or = re.search(
                r"\b[Oo]rientaci[oó]n\s+"
                r"(Norte|Sur|Este|Oeste|"
                r"Noreste|Noroeste|Sureste|Suroeste)\b",
                texto_principal
            )

            if m_or:
                orient_raw = (
                    m_or.group(1)
                )

        resultado[
            "orientacion"
        ] = orient_raw

        # --------------------------------------------------------
        # CERTIFICADO ENERGÉTICO
        # --------------------------------------------------------

        letra, consumo, emision = (
            extraer_certificado_energetico(
                html_principal
            )
        )

        resultado[
            "cert_energia_letra"
        ] = letra

        resultado[
            "cert_energia_consumo"
        ] = consumo

        resultado[
            "cert_energia_emision"
        ] = emision

        # --------------------------------------------------------
        # EXTRAS
        # --------------------------------------------------------

        resultado[
            "extras"
        ] = extraer_extras(
            bloque_principal
        )

        # --------------------------------------------------------
        # SUPERFICIE TERRENO
        # --------------------------------------------------------

        texto_principal = limpiar_texto(
            bloque_principal.get_text(
                " ",
                strip=True
            )
        ) or ""

        m_ter = re.search(
            r"([\d.,]+)\s*m[²2]\s+terreno",
            texto_principal,
            re.IGNORECASE
        )

        if m_ter:
            try:
                v = float(
                    m_ter.group(1)
                    .replace(
                        ".",
                        ""
                    )
                    .replace(
                        ",",
                        "."
                    )
                )

                if 1 <= v <= 100000:
                    resultado[
                        "superficie_terreno_m2"
                    ] = v

            except ValueError:
                pass

        # --------------------------------------------------------
        # NÚMERO DE FOTOS
        # --------------------------------------------------------

        # Primero intentamos en el bloque principal.
        m = re.search(
            r"\b(\d+)\s+[Ff]otos?\b",
            texto_principal
        )

        if m:
            n_fotos = int(
                m.group(1)
            )

            if 1 <= n_fotos <= 200:
                resultado[
                    "num_fotos"
                ] = n_fotos

        # --------------------------------------------------------
        # GPS
        # --------------------------------------------------------

        lat, lon = (
            extraer_coordenadas(
                page_source,
                id_esperado=id_esperado,
                url_esperada=url
            )
        )

        # Si no aparecen, scroll al mapa y reintentar.
        if lat is None:
            try:
                mapa = driver.find_element(
                    By.CSS_SELECTOR,
                    (
                        "[class*='map'], "
                        "[id*='map'], "
                        "[data-testid*='map']"
                    )
                )

                driver.execute_script(
                    "arguments[0].scrollIntoView();",
                    mapa
                )

                time.sleep(
                    2.5
                )

                lat, lon = (
                    extraer_coordenadas(
                        driver.page_source,
                        id_esperado=id_esperado,
                        url_esperada=url
                    )
                )

            except Exception:
                pass

        resultado[
            "latitud"
        ] = lat

        resultado[
            "longitud"
        ] = lon

        # --------------------------------------------------------
        # BAJADA DE PRECIO
        # --------------------------------------------------------

        m_baj = re.search(
            r"[Hh]a\s+bajado\s+"
            r"([\d.,]+)\s*[€Ee]",
            texto_principal
        )

        if m_baj:
            try:
                v = float(
                    m_baj.group(1)
                    .replace(
                        ".",
                        ""
                    )
                    .replace(
                        ",",
                        "."
                    )
                )

                if 100 <= v <= 500000:
                    resultado[
                        "bajada_precio_eur"
                    ] = v

                    resultado[
                        "tiene_bajada_precio"
                    ] = True

            except ValueError:
                pass

        # --------------------------------------------------------
        # REFERENCIA
        # --------------------------------------------------------

        m = re.search(
            r"[Rr]eferencia[:\s]+"
            r"([A-Za-z0-9][A-Za-z0-9\-/]{3,})",
            texto_principal
        )

        if m:
            ref = (
                m.group(1)
                .strip()
            )

            palabras_invalidas = {
                "del",
                "de",
                "las",
                "los",
                "una",
                "que",
                "casa",
                "sobre",
                "para",
                "como",
                "esta",
                "este",
            }

            if (
                len(ref) >= 4
                and
                ref.lower()
                not in palabras_invalidas
                and
                re.search(
                    r"\d",
                    ref
                )
            ):
                resultado[
                    "referencia"
                ] = ref

        # --------------------------------------------------------
        # AGENCIA
        # --------------------------------------------------------

        resultado[
            "agencia"
        ] = extraer_agencia(
            bloque_principal
        )

        # --------------------------------------------------------
        # MARCAR ÉXITO
        # --------------------------------------------------------

        campos_utiles = [
            "descripcion",
            "estado",
            "antiguedad",
            "amueblado",
            "disponibilidad",
            "cert_energia_letra",
            "extras",
            "latitud",
            "orientacion",
            "superficie_terreno_m2",
            "agencia",
        ]

        tiene_datos = any(
            resultado.get(
                campo
            )
            is not None
            for campo in campos_utiles
        )

        resultado[
            "detalle_ok"
        ] = bool(
            resultado[
                "detalle_validado"
            ]
            and
            tiene_datos
        )

    except Exception as e:
        # Si ha muerto Chrome/ChromeDriver no ocultamos el error. El main
        # reiniciará una sesión limpia y reintentará ESTE MISMO anuncio.
        if es_error_driver_grave(e):
            raise

        print(
            f"      AVISO: Error en {url}: {e}"
        )

    return resultado


# ═══════════════════════════════════════════════════════════════
# CARGA DE LOS CSV PROVINCIALES
# ═══════════════════════════════════════════════════════════════

def _canonizar_id_anuncio(valor):
    """Devuelve un ID estable incluso si pandas leyó 123 como 123.0."""

    if valor is None:
        return None

    try:
        if pd.isna(valor):
            return None
    except Exception:
        pass

    texto = str(valor).strip()

    if texto.endswith(".0"):
        texto = texto[:-2]

    return texto or None


def _clave_anuncio(fila):
    """
    Prioriza id_anuncio como clave canónica; si falta, usa el enlace.
    Esto evita duplicados aunque Fotocasa cambie el slug de una URL.
    """

    id_anuncio = _canonizar_id_anuncio(
        fila.get("id_anuncio")
    )

    if id_anuncio:
        return f"id:{id_anuncio}"

    enlace = fila.get("enlace")

    if enlace is None:
        return None

    try:
        if pd.isna(enlace):
            return None
    except Exception:
        pass

    enlace = str(enlace).strip()

    return f"url:{enlace}" if enlace else None


def cargar_datos_provincias():
    """
    Carga todos los CSV provinciales y devuelve un único DataFrame.

    No falla si todavía falta alguna provincia: la muestra en consola y
    procesa las disponibles. Antes de lanzar el TFM definitivo conviene
    comprobar que estén presentes las ocho.
    """

    dataframes = []

    print("\nCARGANDO CSV PROVINCIALES")
    print("─" * 60)

    for slug, nombre_provincia in PROVINCIAS.items():
        ruta = os.path.join(
            PROVINCIAS_DIR,
            f"fotocasa_{slug}.csv"
        )

        if not os.path.exists(ruta):
            print(
                f"AVISO: {nombre_provincia:10s}: no encontrado "
                f"({ruta})"
            )
            continue

        if os.path.getsize(ruta) == 0:
            print(
                f"AVISO: {nombre_provincia:10s}: CSV vacío"
            )
            continue

        try:
            df = pd.read_csv(
                ruta,
                encoding="utf-8-sig"
            )
        except pd.errors.EmptyDataError:
            print(
                f"AVISO: {nombre_provincia:10s}: CSV vacío"
            )
            continue
        except Exception as e:
            print(
                f"ERROR: {nombre_provincia:10s}: error leyendo CSV: {e}"
            )
            continue

        if "enlace" not in df.columns:
            print(
                f"AVISO: {nombre_provincia:10s}: no existe 'enlace'; se omite"
            )
            continue

        originales = len(df)

        df = df[
            df["enlace"].notna()
        ].copy()

        df["provincia"] = nombre_provincia
        df["archivo_origen"] = os.path.basename(ruta)
        df["_clave_anuncio"] = df.apply(
            _clave_anuncio,
            axis=1
        )

        sin_clave = df["_clave_anuncio"].isna().sum()

        if sin_clave:
            df = df[
                df["_clave_anuncio"].notna()
            ].copy()

        antes_dedup = len(df)

        df.drop_duplicates(
            subset=["_clave_anuncio"],
            keep="first",
            inplace=True
        )

        duplicados_locales = antes_dedup - len(df)

        print(
            f"{nombre_provincia:10s}: "
            f"{len(df):6d} anuncios únicos "
            f"| originales={originales} "
            f"| dup={duplicados_locales} "
            f"| sin clave={sin_clave}"
        )

        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    df_total = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False
    )

    antes_global = len(df_total)

    df_total.drop_duplicates(
        subset=["_clave_anuncio"],
        keep="first",
        inplace=True
    )

    df_total.reset_index(
        drop=True,
        inplace=True
    )

    duplicados_globales = antes_global - len(df_total)

    print("─" * 60)
    print(
        f"Registros provinciales cargados : {antes_global}"
    )
    print(
        f"Duplicados entre provincias      : {duplicados_globales}"
    )
    print(
        f"Anuncios únicos a considerar    : {len(df_total)}"
    )

    return df_total


# ═══════════════════════════════════════════════════════════════
# GUARDAR
# ═══════════════════════════════════════════════════════════════

def _guardar(resultados_nuevos):
    """
    Mezcla el lote nuevo con el CSV persistido y deduplica de forma
    estable. Un CSV existente pero vacío se trata como inexistente.
    """

    if not resultados_nuevos:
        return

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    df_nuevos = pd.DataFrame(
        resultados_nuevos
    )

    if "_clave_anuncio" not in df_nuevos.columns:
        df_nuevos["_clave_anuncio"] = df_nuevos.apply(
            _clave_anuncio,
            axis=1
        )

    df_comb = df_nuevos

    if (
        REANUDAR
        and os.path.exists(OUTPUT_CSV)
        and os.path.getsize(OUTPUT_CSV) > 0
    ):
        try:
            df_exist = pd.read_csv(
                OUTPUT_CSV,
                encoding="utf-8-sig"
            )

            if "_clave_anuncio" not in df_exist.columns:
                df_exist["_clave_anuncio"] = df_exist.apply(
                    _clave_anuncio,
                    axis=1
                )

            df_comb = pd.concat(
                [df_exist, df_nuevos],
                ignore_index=True,
                sort=False
            )

        except pd.errors.EmptyDataError:
            df_comb = df_nuevos

    df_comb = df_comb[
        df_comb["_clave_anuncio"].notna()
    ].copy()

    df_comb.drop_duplicates(
        subset=["_clave_anuncio"],
        keep="last",
        inplace=True
    )

    cols_orden = [
        c
        for c in [
            "provincia",
            "municipio",
            "distrito",
            "zona",
            "precio_eur",
        ]
        if c in df_comb.columns
    ]

    if cols_orden:
        df_comb.sort_values(
            by=cols_orden,
            inplace=True,
            na_position="last"
        )

    df_comb.reset_index(
        drop=True,
        inplace=True
    )

    # _clave_anuncio es puramente técnica; no hace falta contaminar el
    # dataset final del TFM con ella.
    df_salida = df_comb.drop(
        columns=["_clave_anuncio"],
        errors="ignore"
    )

    df_salida.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )


# ═══════════════════════════════════════════════════════════════
# GESTIÓN DEL DRIVER EN EJECUCIONES LARGAS
# ═══════════════════════════════════════════════════════════════

def cerrar_driver(driver):
    if driver is None:
        return

    try:
        driver.quit()
    except Exception:
        pass


def iniciar_driver_limpio():
    """Crea Chrome, abre Fotocasa y acepta cookies si aparecen."""

    print("\nIniciando Chrome...")

    driver = crear_driver()

    driver.get(
        "https://www.fotocasa.es/"
    )

    time.sleep(
        random.uniform(2, 4)
    )

    aceptar_cookies(driver)

    return driver


def reiniciar_driver(driver, motivo=None):
    if motivo:
        print(
            f"\nReiniciando Chrome: {motivo}"
        )
    else:
        print(
            "\nReiniciando Chrome..."
        )

    cerrar_driver(driver)

    time.sleep(
        random.uniform(
            PAUSA_REINICIO_MIN,
            PAUSA_REINICIO_MAX
        )
    )

    return iniciar_driver_limpio()


def extraer_detalle_con_reintentos(driver, fila):
    """
    Extrae una ficha. Si Chrome muere, recrea el driver y reintenta el
    MISMO anuncio hasta MAX_REINTENTOS_ANUNCIO veces.

    Devuelve (detalle, driver), porque el driver puede haber cambiado.
    """

    url = fila.get("enlace")
    ultimo_error = None

    for intento in range(
        1,
        MAX_REINTENTOS_ANUNCIO + 1
    ):
        try:
            detalle = extraer_detalle(
                driver=driver,
                url=url,
                id_esperado=fila.get("id_anuncio"),
                titulo_esperado=fila.get("titulo"),
                precio_esperado=fila.get("precio_eur"),
                superficie_esperada=fila.get("superficie_m2"),
                habitaciones_esperadas=fila.get("habitaciones"),
            )

            return detalle, driver

        except Exception as e:
            ultimo_error = e

            if not es_error_driver_grave(e):
                raise

            print(
                f"      Chrome/ChromeDriver ha fallado "
                f"(intento {intento}/{MAX_REINTENTOS_ANUNCIO})"
            )
            print(
                f"         {e}"
            )

            if intento >= MAX_REINTENTOS_ANUNCIO:
                break

            driver = reiniciar_driver(
                driver,
                motivo="sesión inutilizable"
            )

    # No perdemos la fila: queda registrada como fallo técnico y al menos
    # podremos localizarla después. Al reanudar, por defecto esta fila ya
    # figurará en la salida; si se quiere reintentar fallos técnicos habrá
    # que filtrarlos explícitamente.
    detalle_error = {
        "url_final": None,
        "anuncio_disponible": None,
        "motivo_no_disponible": None,
        "titulo_detalle": None,
        "descripcion": None,
        "estado": None,
        "antiguedad": None,
        "antiguedad_min_anos": None,
        "antiguedad_max_anos": None,
        "amueblado": None,
        "disponibilidad": None,
        "cert_energia_letra": None,
        "cert_energia_consumo": None,
        "cert_energia_emision": None,
        "extras": None,
        "num_fotos": None,
        "latitud": None,
        "longitud": None,
        "referencia": None,
        "agencia": None,
        "orientacion": None,
        "superficie_terreno_m2": None,
        "bajada_precio_eur": None,
        "tiene_bajada_precio": False,
        "detalle_ok": False,
        "detalle_validado": False,
        "motivo_validacion": (
            "ERROR_TECNICO_DRIVER: " + str(ultimo_error)
            if ultimo_error
            else "ERROR_TECNICO_DRIVER"
        ),
    }

    # Tras agotar reintentos, dejamos una sesión nueva preparada para la
    # siguiente fila siempre que sea posible.
    try:
        driver = reiniciar_driver(
            driver,
            motivo="agotados los reintentos del anuncio"
        )
    except Exception:
        driver = None

    return detalle_error, driver


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(
        "\nSCRAPER DE DETALLE - FOTOCASA ANDALUCÍA"
    )
    print("─" * 60)

    df_input = cargar_datos_provincias()

    if df_input.empty:
        print(
            "\nERROR: No se ha encontrado ningún anuncio provincial para procesar."
        )
        return

    fin = (
        FILA_FIN
        if FILA_FIN is not None
        else len(df_input)
    )

    df_rango = df_input.iloc[
        FILA_INICIO:fin
    ].copy()

    print(
        f"\nRango: filas {FILA_INICIO}-{fin - 1} "
        f"({len(df_rango)} anuncios)"
    )

    # --------------------------------------------------------
    # REANUDACIÓN
    # --------------------------------------------------------

    ya_procesados = set()

    if (
        REANUDAR
        and os.path.exists(OUTPUT_CSV)
        and os.path.getsize(OUTPUT_CSV) > 0
    ):
        try:
            df_exist = pd.read_csv(
                OUTPUT_CSV,
                encoding="utf-8-sig"
            )

            if "_clave_anuncio" not in df_exist.columns:
                df_exist["_clave_anuncio"] = df_exist.apply(
                    _clave_anuncio,
                    axis=1
                )

            ya_procesados = set(
                df_exist["_clave_anuncio"]
                .dropna()
                .astype(str)
            )

            print(
                f"Ya procesados anteriormente: {len(ya_procesados)}"
            )

        except pd.errors.EmptyDataError:
            print(
                "AVISO: El CSV de salida existe pero está vacío; se empieza desde cero."
            )
            ya_procesados = set()

    pendientes = df_rango[
        ~df_rango["_clave_anuncio"]
        .astype(str)
        .isin(ya_procesados)
    ].copy()

    pendientes.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"   → Pendientes: {len(pendientes)}\n"
    )

    if pendientes.empty:
        print(
            "No hay anuncios pendientes."
        )
        return

    driver = None
    resultados_nuevos = []
    total = len(pendientes)
    procesados_sesion = 0

    try:
        driver = iniciar_driver_limpio()

        for idx, fila in pendientes.iterrows():
            numero = idx + 1
            url = fila.get("enlace")

            if (
                not url
                or pd.isna(url)
            ):
                continue

            # Reinicio preventivo ANTES del siguiente anuncio. Guardamos
            # primero para que incluso una caída durante el reinicio no haga
            # perder el lote actual.
            if (
                procesados_sesion > 0
                and procesados_sesion % REINICIAR_DRIVER_CADA == 0
            ):
                _guardar(
                    resultados_nuevos
                )
                resultados_nuevos = []

                driver = reiniciar_driver(
                    driver,
                    motivo=(
                        f"reinicio preventivo cada "
                        f"{REINICIAR_DRIVER_CADA} anuncios"
                    )
                )

            if driver is None:
                driver = iniciar_driver_limpio()

            id_an = fila.get(
                "id_anuncio",
                "?"
            )

            provincia_str = (
                fila.get("provincia")
                or "?"
            )

            zona_str = (
                fila.get("municipio")
                or fila.get("ciudad")
                or fila.get("distrito")
                or fila.get("zona")
                or "?"
            )

            print(
                f"[{numero:5d}/{total}] "
                f"#{id_an} | {provincia_str} | {zona_str}"
            )

            detalle, driver = extraer_detalle_con_reintentos(
                driver,
                fila
            )

            fila_dict = fila.to_dict()
            fila_dict.pop("_clave_anuncio", None)
            fila_dict.update(detalle)

            resultados_nuevos.append(
                fila_dict
            )

            procesados_sesion += 1

            icon = (
                "✓"
                if detalle.get("detalle_ok")
                else "✗"
            )

            valid_icon = (
                "✓"
                if detalle.get("detalle_validado")
                else "✗"
            )

            disponible_valor = detalle.get(
                "anuncio_disponible"
            )

            if disponible_valor is True:
                disponible_str = "Sí"
            elif disponible_valor is False:
                disponible_str = "No"
            else:
                disponible_str = "N/D"

            lat = detalle.get("latitud")
            lon = detalle.get("longitud")

            lat_str = (
                f"{lat:.4f}"
                if lat is not None
                else "N/D"
            )

            lon_str = (
                f"{lon:.4f}"
                if lon is not None
                else "N/D"
            )

            cert = detalle.get("cert_energia_letra") or "N/D"
            cons = str(detalle.get("cert_energia_consumo") or "N/D")
            emis = str(detalle.get("cert_energia_emision") or "N/D")

            print(
                f"       {icon} "
                f"Disponible:{disponible_str} | "
                f"Valid:{valid_icon} | "
                f"Estado:{str(detalle.get('estado') or 'N/D'):12s} | "
                f"Antigüedad:{str(detalle.get('antiguedad') or 'N/D'):15s} | "
                f"Cert:{cert} kWh:{cons} CO₂:{emis} | "
                f"GPS:{lat_str},{lon_str}"
            )

            print(
                "       -> "
                f"{detalle.get('motivo_validacion')}"
            )

            if numero % CHECKPOINT_CADA == 0:
                _guardar(
                    resultados_nuevos
                )
                resultados_nuevos = []

                print(
                    f"   Checkpoint ({numero}/{total})"
                )

            time.sleep(
                random.uniform(
                    PAUSA_MIN,
                    PAUSA_MAX
                )
            )

    except KeyboardInterrupt:
        print(
            "\n\nInterrupción manual detectada. Guardando antes de salir..."
        )
        _guardar(
            resultados_nuevos
        )
        resultados_nuevos = []
        print(
            "Progreso guardado. Puedes reanudar ejecutando el script otra vez."
        )

    finally:
        # Guardado defensivo incluso si aparece una excepción inesperada.
        _guardar(
            resultados_nuevos
        )
        cerrar_driver(driver)

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    print("\n" + "═" * 60)
    print("RESULTADO ACTUAL")
    print("═" * 60)

    if (
        os.path.exists(OUTPUT_CSV)
        and os.path.getsize(OUTPUT_CSV) > 0
    ):
        try:
            df_f = pd.read_csv(
                OUTPUT_CSV,
                encoding="utf-8-sig"
            )
        except pd.errors.EmptyDataError:
            print(
                "AVISO: El CSV de salida está vacío."
            )
            return

        n = len(df_f)

        def pct(col):
            if col not in df_f.columns:
                return "N/D"

            v = df_f[col].notna().sum()

            return (
                f"{v:6d} ({v / n * 100:5.1f}%)"
                if n
                else "0"
            )

        print(
            f"Total anuncios guardados     : {n}"
        )

        if "provincia" in df_f.columns:
            print("\nPOR PROVINCIA")
            print(
                df_f["provincia"]
                .value_counts(dropna=False)
                .to_string()
            )

        if "anuncio_disponible" in df_f.columns:
            disponibles = (
                df_f["anuncio_disponible"]
                .eq(True)
                .sum()
            )
            retirados = (
                df_f["anuncio_disponible"]
                .eq(False)
                .sum()
            )

            print(
                f"\nAnuncio disponible = True  : {int(disponibles):6d}"
            )
            print(
                f"Anuncio retirado = True     : {int(retirados):6d}"
            )

        if "detalle_validado" in df_f.columns:
            print(
                "detalle_validado = True     : "
                f"{int(df_f['detalle_validado'].fillna(False).sum()):6d}"
            )

        if "detalle_ok" in df_f.columns:
            print(
                "detalle_ok = True           : "
                f"{int(df_f['detalle_ok'].fillna(False).sum()):6d}"
            )

        if "motivo_validacion" in df_f.columns:
            errores_driver = (
                df_f["motivo_validacion"]
                .fillna("")
                .astype(str)
                .str.startswith("ERROR_TECNICO_DRIVER")
                .sum()
            )

            print(
                f"Errores técnicos de driver   : {int(errores_driver):6d}"
            )

        print(
            f"Con GPS                      : {pct('latitud')}"
        )
        print(
            f"Con estado                   : {pct('estado')}"
        )
        print(
            f"Con antigüedad               : {pct('antiguedad')}"
        )
        print(
            f"Con cert. letra              : {pct('cert_energia_letra')}"
        )
        print(
            f"Con cert. consumo (kWh)      : {pct('cert_energia_consumo')}"
        )
        print(
            f"Con cert. emisión (CO₂)      : {pct('cert_energia_emision')}"
        )
        print(
            f"Con descripción              : {pct('descripcion')}"
        )

        print(
            f"\n{OUTPUT_CSV}"
        )

        columnas_muestra = [
            c
            for c in [
                "provincia",
                "municipio",
                "id_anuncio",
                "titulo",
                "url_final",
                "anuncio_disponible",
                "motivo_no_disponible",
                "titulo_detalle",
                "precio_eur",
                "detalle_validado",
                "motivo_validacion",
                "estado",
                "antiguedad",
                "cert_energia_letra",
                "cert_energia_consumo",
                "cert_energia_emision",
                "latitud",
                "longitud",
            ]
            if c in df_f.columns
        ]

        print("\nMuestra:")
        print(
            df_f[columnas_muestra]
            .head(10)
            .to_string()
        )


if __name__ == "__main__":
    main()
