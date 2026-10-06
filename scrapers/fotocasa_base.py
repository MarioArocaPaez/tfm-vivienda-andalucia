
    
  
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
# Umbral conservador para reducir el riesgo de topes del listado.
UMBRAL_DIVIDIR_ZONAS = 350
MAX_NIVEL_ZONAS = 2

PAUSA_BUSQUEDA_MIN = 2.0
PAUSA_BUSQUEDA_MAX = 4.0
PAUSA_MUNICIPIO_MIN = 4.0
PAUSA_MUNICIPIO_MAX = 8.0

MUNICIPIOS_POR_DRIVER = 5
MAX_REINTENTOS_DRIVER = 4

# Reescanea también las búsquedas marcadas como completadas.
# Esto permite capturar anuncios nuevos en ejecuciones posteriores.
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

