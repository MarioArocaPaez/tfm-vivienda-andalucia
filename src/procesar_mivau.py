"""
Convierte las tablas Excel del Boletín Online del MIVAU en CSV ordenados (Andalucía).

Tablas que se esperan en la carpeta de entrada (descargadas de
https://apps.fomento.gob.es/BoletinOnline2/?nivel=2&orden=34000000):
    2.3    Transacciones inmobiliarias de vivienda nueva por municipios
    2.4    Transacciones inmobiliarias de vivienda de segunda mano por municipios
    3.3.1  Valor medio de las transacciones de vivienda libre nueva
    3.3.2  Valor medio de las transacciones de vivienda libre de segunda mano
Cada fichero se identifica por el número de tabla de su título, no por el
nombre del fichero (el MIVAU llega a dar el mismo nombre a tablas distintas).

Uso (desde la raíz del repositorio):
    python src/procesar_mivau.py [carpeta_excels] [carpeta_salida]
Por defecto: data/externos/mivau  ->  data/procesado

Salida:
    mivau_transacciones_municipios_andalucia.csv  (municipio, trimestre, tipo, n_transacciones)
    mivau_valor_provincias_andalucia.csv          (provincia, trimestre, métrica, valor)
"""

import glob
import os
import re
import sys
import unicodedata

import pandas as pd
import xlrd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "data", "externos", "mivau")
SALIDA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RAIZ, "data", "procesado")

PROVINCIAS = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"]

TABLAS_MUNICIPALES = {"2.3": "nueva", "2.4": "segunda_mano"}
TABLAS_PROVINCIALES = {
    "3.3.1": "valor_medio_libre_nueva_eur",
    "3.3.2": "valor_medio_libre_segunda_mano_eur",
    "3.1.1": "valor_total_libre_nueva_miles_eur",
    "3.1.2": "valor_total_libre_segunda_mano_miles_eur",
}


def norm(texto):
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", texto.lower().strip())


PROV_NORM = {norm(p) for p in PROVINCIAS}


def numero_tabla(libro):
    hoja = libro.sheet_by_index(0)
    titulo = " ".join(str(v) for r in range(min(10, hoja.nrows)) for v in hoja.row_values(r))
    m = re.search(r"Tabla (\d+(?:\.\d+)*)", titulo)
    return m.group(1) if m else None


def columnas_periodo(hoja, fila_anio, fila_trim, col_inicio):
    """[(columna, año, trimestre)] a partir de las dos filas de cabecera."""
    salida, anio = [], None
    for c in range(col_inicio, hoja.ncols):
        m = re.search(r"(20\d\d)", str(hoja.cell_value(fila_anio, c)))
        if m:
            anio = int(m.group(1))
        mt = re.search(r"([1-4])", str(hoja.cell_value(fila_trim, c)))
        if anio and mt:
            salida.append((c, anio, int(mt.group(1))))
    return salida


def leer_municipal(libro, tipo):
    hoja = libro.sheet_by_index(0)
    periodos = columnas_periodo(hoja, 10, 12, 3)
    nombres = [str(hoja.cell_value(r, 1)).strip() for r in range(hoja.nrows)]
    filas, ccaa, provincia = [], None, None

    for r in range(13, hoja.nrows):
        nombre = nombres[r]
        if not nombre or nombre.startswith("*"):
            continue
        if nombre.isupper():                      # cabecera de comunidad autónoma
            ccaa, provincia = nombre, None
            continue
        siguiente = next((nombres[k] for k in range(r + 1, min(r + 5, hoja.nrows)) if nombres[k]), "")
        # Cabecera de provincia: nombre de provincia seguido de un "reinicio" del
        # orden alfabético (la capital, en cambio, va seguida de un nombre posterior).
        # Según el fichero, la fila de cabecera viene vacía o con valores.
        if provincia is None or (norm(nombre) in PROV_NORM and norm(siguiente) < norm(nombre)):
            provincia = nombre
            continue
        for c, anio, trim in periodos:
            v = hoja.cell_value(r, c)
            filas.append(dict(ccaa=ccaa, provincia=provincia, municipio=nombre, anio=anio,
                              trimestre=trim, tipo=tipo,
                              n_transacciones=float(v) if v != "" else None))
    return pd.DataFrame(filas)


def leer_provincial(libro, metrica):
    filas = []
    for hoja in libro.sheets():                   # una pestaña por bloque de años
        fila_anio = next(r for r in range(6, 16)
                         if any(re.search(r"Año 20\d\d", str(v)) for v in hoja.row_values(r)))
        fila_trim = next(r for r in range(fila_anio + 1, fila_anio + 4)
                         if sum(bool(re.search(r"[1-4]", str(v))) for v in hoja.row_values(r)) >= 4)
        periodos = columnas_periodo(hoja, fila_anio, fila_trim, 2)
        for r in range(fila_trim + 1, hoja.nrows):
            nombre = str(hoja.cell_value(r, 1)).strip()
            if norm(nombre) in PROV_NORM or norm(nombre) == "andalucia":
                for c, anio, trim in periodos:
                    v = hoja.cell_value(r, c)
                    if v != "":
                        filas.append(dict(ambito=nombre, anio=anio, trimestre=trim,
                                          metrica=metrica, valor=float(v)))
    return pd.DataFrame(filas).drop_duplicates(["ambito", "anio", "trimestre"])


def main():
    os.makedirs(SALIDA, exist_ok=True)
    municipales, provinciales = [], []
    for ruta in sorted(glob.glob(os.path.join(ENTRADA, "*.xls")) + glob.glob(os.path.join(ENTRADA, "*.XLS"))):
        libro = xlrd.open_workbook(ruta)
        tabla = numero_tabla(libro)
        if tabla in TABLAS_MUNICIPALES:
            municipales.append(leer_municipal(libro, TABLAS_MUNICIPALES[tabla]))
        elif tabla in TABLAS_PROVINCIALES:
            provinciales.append(leer_provincial(libro, TABLAS_PROVINCIALES[tabla]))
        else:
            print(f"Se ignora {os.path.basename(ruta)} (tabla {tabla})")

    if municipales:
        m = pd.concat(municipales)
        m = m[m.ccaa == "ANDALUCÍA"]
        m.to_csv(os.path.join(SALIDA, "mivau_transacciones_municipios_andalucia.csv"),
                 index=False, encoding="utf-8-sig")
        print(f"Municipal: {m.municipio.nunique()} municipios | "
              f"{m.groupby('provincia').municipio.nunique().to_dict()}")
    if provinciales:
        p = pd.concat(provinciales).drop_duplicates(["ambito", "anio", "trimestre", "metrica"])
        p.to_csv(os.path.join(SALIDA, "mivau_valor_provincias_andalucia.csv"),
                 index=False, encoding="utf-8-sig")
        print(f"Provincial: {p.groupby('metrica').size().to_dict()}")


if __name__ == "__main__":
    main()
