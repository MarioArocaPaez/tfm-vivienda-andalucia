"""
Variables adicionales para atacar el sesgo en los extremos de precio.

El análisis de errores del modelo base mostró regresión a la media: las viviendas
más baratas se sobrevaloran (+13,6 % en el quintil 1) y las más caras se
infravaloran (-14,3 % en el quintil 5). Los atributos estructurados del anuncio no
recogen lo que separa una vivienda de lujo en primera línea de playa de una a
reformar en el interior. Este módulo añade tres grupos de variables:

1. "texto": indicadores binarios extraídos del título y la descripción con
   expresiones regulares (vistas al mar, primera línea, a reformar, lujo...).
   Las viviendas ocupadas o sin posesión que solo se detectan por el texto se
   eliminan en la limpieza (src/limpieza.py), igual que las marcadas en los campos.
   Los términos se eligieron explorando los residuos fuera de muestra (OOF) del
   modelo base SOLO en el conjunto de entrenamiento; el de prueba no se miró.
2. "geo": distancia a la costa y a la capital de provincia, calculadas desde las
   coordenadas del anuncio.
3. "renta": renta neta media por persona del municipio (INE, Atlas de
   Distribución de Renta de los Hogares).

Cada grupo se puede activar por separado en preprocesado.preparar(grupos=...),
lo que permite medir su aportación con una ablación.
"""

import glob
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSTA = os.path.join(RAIZ, "data", "externos", "geo", "costa_andalucia.geojson")
DIR_RENTA = os.path.join(RAIZ, "data", "externos", "ine", "renta")

# ----------------------------------------------------------------------
# 1. Texto
# ----------------------------------------------------------------------
# Se buscan sobre título + descripción en minúsculas y sin tildes.
PATRONES_TEXTO = {
    "txt_vistas_mar": r"vistas? (?:al|a el|panoramicas? al|despejadas? al) mar|vistas al oceano|sea views?",
    "txt_primera_linea": r"primera linea",
    "txt_playa": r"\bplaya",
    "txt_lujo": r"\blujo|luxury|alto standing|alta gama|altas calidades",
    "txt_piscina_desbordante": r"infinity|desbordante",
    "txt_golf": r"\bgolf",
    "txt_spa": r"\bsauna|\bspa\b|jacuzzi|gimnasio",
    "txt_atico": r"\batico",
    "txt_a_estrenar": r"a estrenar",
    "txt_reformado": r"reformad[oa]|recien reformad",
    "txt_a_reformar": r"a reformar|para reformar|necesita reforma|reforma integral|para rehabilitar|a rehabilitar",
    "txt_cueva": r"\bcueva",
    "txt_inversion": r"inversion|inversores|rentabilidad",
}
VARIABLES_TEXTO = list(PATRONES_TEXTO)


def normalizar_texto(serie):
    def norm(s):
        s = unicodedata.normalize("NFKD", str(s).lower())
        return "".join(c for c in s if not unicodedata.combining(c))
    return serie.fillna("").map(norm)


def variables_texto(df):
    texto = normalizar_texto(df["titulo"].fillna("") + " " + df["descripcion"].fillna(""))
    return pd.DataFrame({nombre: texto.str.contains(patron, regex=True).astype(int)
                         for nombre, patron in PATRONES_TEXTO.items()}, index=df.index)


# ----------------------------------------------------------------------
# 2. Geografía
# ----------------------------------------------------------------------
# Ayuntamiento de cada capital (aprox.), para la distancia a la capital.
CAPITALES = {
    "Almería": (36.8381, -2.4597), "Cádiz": (36.5297, -6.2925), "Córdoba": (37.8882, -4.7794),
    "Granada": (37.1773, -3.5986), "Huelva": (37.2614, -6.9447), "Jaén": (37.7796, -3.7849),
    "Málaga": (36.7213, -4.4214), "Sevilla": (37.3891, -5.9845),
}
VARIABLES_GEO = ["dist_costa_km", "dist_capital_km"]

# Proyección equirrectangular centrada en Andalucía (lat 37º): a esta escala
# el error frente a la distancia geodésica es inferior al 1 %.
_KM_LAT = 110.57
_KM_LON = 111.32 * np.cos(np.radians(37.0))


def _a_km(lon, lat):
    return np.asarray(lon) * _KM_LON, np.asarray(lat) * _KM_LAT


def _cargar_costa():
    import shapely
    from shapely.geometry import shape
    with open(COSTA, encoding="utf-8") as f:
        geom = shape(json.load(f)["geometry"])
    return shapely.transform(geom, lambda c: np.column_stack(_a_km(c[:, 0], c[:, 1])))


def variables_geo(df):
    from shapely import points, distance
    costa = _cargar_costa()
    x, y = _a_km(df["longitud"], df["latitud"])
    validas = ~(np.isnan(x) | np.isnan(y))
    dist_costa = np.full(len(df), np.nan)
    dist_costa[validas] = distance(points(x[validas], y[validas]), costa)

    cap = df["provincia"].map(CAPITALES)
    cap_lat = cap.map(lambda c: c[0] if isinstance(c, tuple) else np.nan)
    cap_lon = cap.map(lambda c: c[1] if isinstance(c, tuple) else np.nan)
    cx, cy = _a_km(cap_lon, cap_lat)
    dist_capital = np.hypot(x - cx, y - cy)
    return pd.DataFrame({"dist_costa_km": dist_costa, "dist_capital_km": dist_capital}, index=df.index)


# ----------------------------------------------------------------------
# 3. Renta municipal (INE, ADRH)
# ----------------------------------------------------------------------
VARIABLES_RENTA = ["renta_neta_persona"]
INDICADOR_RENTA = "renta neta media por persona"


def _numero(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def cargar_renta(directorio=DIR_RENTA):
    """Lee los ficheros del ADRH (uno por provincia, CSV ';' o xlsx descargados de
    INEbase) y devuelve cod_ine -> renta neta media por persona del último año.

    Se queda con las filas de ámbito municipal (sin distrito ni sección).
    """
    ficheros = sorted(glob.glob(os.path.join(directorio, "*.csv")) +
                      glob.glob(os.path.join(directorio, "*.xlsx")))
    if not ficheros:
        raise FileNotFoundError(f"No hay ficheros de renta del INE en {directorio} (ver README).")
    partes = []
    for ruta in ficheros:
        if ruta.endswith(".csv"):
            # INEbase exporta con ";" o tabulador, en UTF-8 o Latin-1 según la opción elegida
            for codificacion in ("utf-8-sig", "latin-1"):
                try:
                    with open(ruta, encoding=codificacion) as f:
                        cabecera = f.readline()
                    sep = ";" if cabecera.count(";") >= cabecera.count("\t") else "\t"
                    t = pd.read_csv(ruta, sep=sep, dtype=str, encoding=codificacion)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            t = pd.read_excel(ruta, dtype=str)
        t.columns = [normalizar_texto(pd.Series([c]))[0].strip() for c in t.columns]
        col_mun = next((c for c in t.columns if c.startswith("municipio")), None)
        if col_mun is None:
            raise ValueError(f"{os.path.basename(ruta)} no tiene columna de municipios: "
                             "hay que descargar la tabla provincial con detalle municipal.")
        col_ind = next(c for c in t.columns if "indicador" in c)
        col_valor = "total" if "total" in t.columns else t.columns[-1]
        filas = t[normalizar_texto(t[col_ind]).str.strip().eq(INDICADOR_RENTA)]
        for c in ("distritos", "secciones"):
            if c in t.columns:
                filas = filas[filas[c].isna() | filas[c].str.strip().eq("")]
        filas = filas.assign(cod_ine=filas[col_mun].str.extract(r"^(\d{5})")[0],
                             periodo=pd.to_numeric(filas["periodo"], errors="coerce"),
                             valor=filas[col_valor].map(_numero)).dropna(subset=["cod_ine", "valor"])
        partes.append(filas[["cod_ine", "periodo", "valor"]])
    renta = pd.concat(partes)
    ultimo = renta["periodo"].max()
    renta = renta[renta["periodo"] == ultimo].drop_duplicates("cod_ine")
    renta["cod_ine"] = renta["cod_ine"].astype(int)
    return renta.set_index("cod_ine")["valor"].rename("renta_neta_persona"), int(ultimo)


PROVINCIAS_ANDALUCIA = ("04", "11", "14", "18", "21", "23", "29", "41")


def extraer_andalucia(ruta_nacional, destino=os.path.join(DIR_RENTA, "adrh_andalucia_municipios.csv")):
    """Reduce el fichero nacional del ADRH (tabla 30824, ~400 MB) a las filas de
    ámbito municipal de Andalucía, con el mismo formato, para no guardar el original."""
    t = pd.read_csv(ruta_nacional, sep=";", dtype=str, encoding="utf-8-sig")
    municipal = t["Distritos"].isna() & t["Secciones"].isna()
    andalucia = t["Municipios"].str[:2].isin(PROVINCIAS_ANDALUCIA)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    t[municipal & andalucia].to_csv(destino, sep=";", index=False, encoding="utf-8-sig")
    return destino


def variables_renta(df):
    renta, _ = cargar_renta()
    cod = pd.to_numeric(df["cod_ine"], errors="coerce")
    return pd.DataFrame({"renta_neta_persona": cod.map(renta)}, index=df.index)


GRUPOS = {"texto": VARIABLES_TEXTO, "geo": VARIABLES_GEO, "renta": VARIABLES_RENTA}
_FUNCIONES = {"texto": variables_texto, "geo": variables_geo, "renta": variables_renta}


def anadir(df, grupos):
    """Añade al DataFrame las columnas de los grupos pedidos."""
    for g in grupos:
        df = df.join(_FUNCIONES[g](df))
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--extraer-renta":
        # python src/variables_extra.py --extraer-renta ruta/30824.csv
        print("Extracto guardado en", extraer_andalucia(sys.argv[2]))
        sys.exit()
    from preprocesado import cargar_datos
    df = cargar_datos()
    print((variables_texto(df).mean() * 100).round(1).to_string())
    print(variables_geo(df).describe().round(1))
    try:
        r = variables_renta(df)
        print(f"Renta: cobertura {r['renta_neta_persona'].notna().mean():.1%}")
        print(r.describe().round(0))
    except FileNotFoundError as e:
        print(e)
