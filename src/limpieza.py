"""
Limpieza del conjunto de datos de Fotocasa enriquecido.

Parte de la salida de enriquecer_fotocasa.py y produce el conjunto de datos de
análisis: un anuncio por vivienda, con valores plausibles y variables en un
formato utilizable por los modelos.

Uso (desde la raíz del repositorio):
    python src/limpieza.py [entrada.csv] [carpeta_salida]
Por defecto:
    data/procesado/fotocasa_andalucia_enriquecido.csv  ->  data/procesado

Salida:
    fotocasa_andalucia_limpio.csv   conjunto de datos de análisis
    informe_limpieza.csv            anuncios eliminados por cada regla

Las reglas se aplican en orden y cada una se documenta en REGLAS. Los umbrales
están agrupados en UMBRALES para poder justificarlos (y cambiarlos) en un único
sitio.
"""

import os
import re
import sys

import unicodedata

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESADO = os.path.join(RAIZ, "data", "procesado")
ENTRADA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROCESADO, "fotocasa_andalucia_enriquecido.csv")
SALIDA = sys.argv[2] if len(sys.argv) > 2 else PROCESADO

UMBRALES = {
    "precio_min": 20_000,         # por debajo: plazas de garaje, cuotas, errores de precio
    "precio_max": 5_000_000,      # por encima: mansiones singulares, muy poco representadas
    "superficie_min": 15,         # m² mínimos de una vivienda habitable
    "superficie_max": 1_500,      # por encima: superficie de parcela introducida como construida
    "precio_m2_min": 200,         # incoherencia precio/superficie
    "precio_m2_max": 20_000,
    "habitaciones_max": 15,
    "banos_max": 10,
}

# Situaciones en las que el precio no refleja el mercado libre de compraventa.
DISPONIBILIDAD_EXCLUIDA = {"Subasta", "Nuda propiedad", "Ocupado"}

# La misma situación cuando solo consta en el texto del anuncio (Fotocasa no la
# marca en el campo de disponibilidad). Se busca en título + descripción en
# minúsculas y sin tildes. Revisadas a mano las coincidencias: no hay negaciones
# del tipo "no está ocupada".
PATRON_OCUPADA = (r"\bokupa|ocupad[ao]s? ilegal|sin posesion|vivienda ocupada|inmueble ocupado"
                  r"|esta ocupad")

CAPITALES = {"Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"}

# Equipamiento: el listado y la ficha de detalle informan de forma incompleta y
# distinta, así que una característica se considera presente si aparece en
# cualquiera de las dos fuentes.
EQUIPAMIENTO = {   # columna del listado -> etiqueta en "extras" del detalle
    "ascensor": "ascensor", "terraza": "terraza", "balcon": "balcón", "parking": "garaje",
    "piscina": "piscina", "jardin": "jardín", "aire_acondicionado": "aire acondicionado",
    "calefaccion": "calefacción", "trastero": "trastero",
}
EXTRAS_SOLO_DETALLE = {
    "cocina_equipada": "cocina equipada", "patio": "patio", "puerta_blindada": "puerta blindada",
    "electrodomesticos": "electrodomésticos", "domotica": "domótica",
}

ESTADOS = ["A reformar", "Bien", "Muy bien", "Reformado", "Casi nuevo"]
ANTIGUEDADES = ["Menos de 1 año", "1 a 5 años", "5 a 10 años", "10 a 20 años", "20 a 30 años",
                "30 a 50 años", "50 a 70 años", "70 a 100 años", "+ 100 años"]

COLUMNAS_SALIDA = [
    # identificación
    "id_anuncio", "enlace", "fecha_extraccion", "origen_extraccion",
    # ubicación
    "provincia", "municipio", "municipio_ine", "cod_ine", "es_capital", "zona", "latitud", "longitud",
    # variable objetivo
    "precio_eur", "log_precio",
    # características principales
    "tipo_vivienda", "superficie_m2", "habitaciones", "banos", "planta_num", "superficie_terreno_m2",
    # estado y situación
    "estado", "antiguedad", "antiguedad_anos", "obra_nueva", "amueblado", "orientacion", "alquilado",
    # equipamiento
    *EQUIPAMIENTO, *EXTRAS_SOLO_DETALLE,
    # anuncio
    "num_fotos", "tiene_bajada_precio", "longitud_descripcion", "titulo", "descripcion",
    # contexto municipal y provincial (INE, MIVAU)
    "poblacion_2025", "mivau_trans_total_4t", "mivau_trans_por_1000hab",
    "mivau_valor_medio_ref_eur", "mivau_ref_tipo",
    # informativas: derivadas del precio, NO usar como predictoras
    "precio_m2", "bajada_precio_eur",
]


# ------------------------------------------------------------------
# Normalización de variables
# ------------------------------------------------------------------

def a_bool(serie):
    return serie.astype(str).str.strip().str.lower().isin({"true", "1", "sí", "si", "yes"})


def tipo_desde_titulo(titulo):
    """Tipo de vivienda para anuncios sin tipo (sobre todo promociones de obra nueva)."""
    t = str(titulo).lower()
    for clave, tipo in [("ático", "ático"), ("atico", "ático"), ("dúplex", "dúplex"), ("duplex", "dúplex"),
                        ("loft", "estudio"), ("estudio", "estudio"), ("apartamento", "apartamento"),
                        ("chalet", "casa"), ("adosad", "casa"), ("casa", "casa"), ("villa", "casa"),
                        ("planta baja", "planta baja"), ("piso", "piso")]:
        if clave in t:
            return tipo
    return "sin especificar"


def normalizar_tipo(df):
    tipo = df["tipo_vivienda"].replace({"villa": "casa"})
    return tipo.where(tipo.notna(), df["titulo"].map(tipo_desde_titulo))


def planta_a_numero(valor):
    """'2ª planta' -> 2, 'bajo' -> 0, 'sótano' -> -1. Cualquier otra cosa -> NaN.

    El campo de Fotocasa mezcla a veces trozos de la descripción, y "+15 planta"
    es un valor de relleno del portal (aparece en pisos de edificios bajos), así
    que solo se aceptan formas inequívocas."""
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().lower()
    if v in {"bajo", "bajos", "planta baja", "entresuelo"}:
        return 0
    if v in {"sótano", "semisótano"}:
        return -1
    m = re.fullmatch(r"(\d{1,2})ª? planta", v)
    if m and v != "15 planta" and int(m.group(1)) <= 30:
        return int(m.group(1))
    return np.nan


def antiguedad_media(df):
    """Punto medio del intervalo de antigüedad (años); '+ 100 años' -> 100."""
    mn, mx = df["antiguedad_min_anos"], df["antiguedad_max_anos"]
    media = (mn + mx) / 2
    media = media.where(df["antiguedad"] != "Menos de 1 año", 0.5)
    return media.where(df["antiguedad"].isin(ANTIGUEDADES))


def construir_variables(df):
    extras = df["extras"].fillna("").str.lower()
    for col, etiqueta in EQUIPAMIENTO.items():
        df[col] = a_bool(df[col]) | extras.str.contains(etiqueta, regex=False)
    for col, etiqueta in EXTRAS_SOLO_DETALLE.items():
        df[col] = extras.str.contains(etiqueta, regex=False)

    df["tipo_vivienda"] = normalizar_tipo(df)
    df["planta_num"] = df["planta"].map(planta_a_numero)
    df["estado"] = df["estado"].where(df["estado"].isin(ESTADOS), "Desconocido")
    df["antiguedad_anos"] = antiguedad_media(df)
    df["antiguedad"] = df["antiguedad"].where(df["antiguedad"].isin(ANTIGUEDADES), "Desconocida")
    df["orientacion"] = df["orientacion"].fillna("Desconocida")
    df["obra_nueva"] = (a_bool(df["obra_nueva"]) | (df["antiguedad"] == "Menos de 1 año")
                        | df["titulo"].astype(str).str.contains("Obra Nueva", case=False))
    df["amueblado"] = df["amueblado"].astype(str).str.strip().eq("Sí")
    df["alquilado"] = df["disponibilidad"].astype(str).str.startswith("Alquilado")
    df["tiene_bajada_precio"] = a_bool(df["tiene_bajada_precio"])
    df["es_capital"] = df["municipio_ine"].isin(CAPITALES) & (df["municipio_ine"] == df["provincia"])
    df["longitud_descripcion"] = df["descripcion"].fillna("").str.len()
    df["precio_m2"] = (df["precio_eur"] / df["superficie_m2"]).round(2)
    df["log_precio"] = np.log(df["precio_eur"])
    # La superficie de parcela solo tiene sentido en viviendas unifamiliares.
    df.loc[df["tipo_vivienda"] != "casa", "superficie_terreno_m2"] = np.nan
    return df


# ------------------------------------------------------------------
# Reglas de filtrado (se aplican en este orden)
# ------------------------------------------------------------------

def texto_normalizado(df):
    texto = (df["titulo"].fillna("") + " " + df["descripcion"].fillna("")).str.lower()
    return texto.map(lambda s: "".join(c for c in unicodedata.normalize("NFKD", s)
                                       if not unicodedata.combining(c)))


def clave_duplicado(df):
    desc = df["descripcion"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str[:200]
    campos = df[["municipio", "precio_eur", "superficie_m2", "habitaciones", "banos"]].apply(lambda c: c.map(str))
    return campos.agg("|".join, axis=1) + "|" + desc


U = UMBRALES
REGLAS = [
    ("detalle no validado",
     "Solo anuncios cuya ficha de detalle se ha podido validar (criterio acordado con el tutor).",
     lambda d: a_bool(d["detalle_validado"])),
    ("municipio no identificado",
     "Sin código INE: pseudo-municipio de navegación de Fotocasa.",
     lambda d: d["cod_ine"].notna()),
    ("finca rústica",
     "Fuera del alcance: vivienda rústica con superficies y precios no comparables.",
     lambda d: d["tipo_vivienda"].astype(str) != "finca"),
    ("disponibilidad no de mercado",
     f"Precio no comparable: {', '.join(sorted(DISPONIBILIDAD_EXCLUIDA))}.",
     lambda d: ~d["disponibilidad"].isin(DISPONIBILIDAD_EXCLUIDA)),
    ("ocupada según el texto",
     "Vivienda ocupada o sin posesión que solo se indica en la descripción; mismo criterio "
     "que la regla anterior.",
     lambda d: ~texto_normalizado(d).str.contains(PATRON_OCUPADA, regex=True)),
    ("sin precio o superficie",
     "Faltan la variable objetivo o la superficie.",
     lambda d: d["precio_eur"].notna() & d["superficie_m2"].notna()),
    ("precio fuera de rango",
     f"Precio fuera de [{U['precio_min']:,}, {U['precio_max']:,}] €.",
     lambda d: d["precio_eur"].between(U["precio_min"], U["precio_max"])),
    ("superficie fuera de rango",
     f"Superficie fuera de [{U['superficie_min']}, {U['superficie_max']:,}] m².",
     lambda d: d["superficie_m2"].between(U["superficie_min"], U["superficie_max"])),
    ("precio/m² incoherente",
     f"Precio por m² fuera de [{U['precio_m2_min']}, {U['precio_m2_max']:,}] €/m².",
     lambda d: (d["precio_eur"] / d["superficie_m2"]).between(U["precio_m2_min"], U["precio_m2_max"])),
    ("habitaciones o baños imposibles",
     f"Más de {U['habitaciones_max']} habitaciones o {U['banos_max']} baños.",
     lambda d: ~((d["habitaciones"] > U["habitaciones_max"]) | (d["banos"] > U["banos_max"]))),
    ("anuncio duplicado",
     "Misma vivienda publicada varias veces: coinciden municipio, precio, superficie, "
     "habitaciones, baños y descripción. Se conserva la extracción más reciente.",
     lambda d: ~clave_duplicado(d).duplicated(keep="first")),
]


def main():
    df = pd.read_csv(ENTRADA, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["fecha_extraccion", "id_anuncio"], ascending=[False, True]).reset_index(drop=True)
    df["tipo_vivienda"] = df["tipo_vivienda"].replace({"villa": "casa"})

    informe = [dict(paso=0, regla="entrada", descripcion=os.path.basename(ENTRADA),
                    eliminados=0, restantes=len(df))]
    print(f"Entrada: {len(df):,} anuncios")
    for i, (nombre, descripcion, condicion) in enumerate(REGLAS, 1):
        mantener = condicion(df).fillna(False).astype(bool)
        eliminados = int((~mantener).sum())
        df = df[mantener].copy()
        informe.append(dict(paso=i, regla=nombre, descripcion=descripcion,
                            eliminados=eliminados, restantes=len(df)))
        print(f"  {i:2d}. {nombre:34s} -{eliminados:6,d}  -> {len(df):7,d}")

    df = construir_variables(df)
    df = df[COLUMNAS_SALIDA].sort_values(["provincia", "municipio", "id_anuncio"])

    os.makedirs(SALIDA, exist_ok=True)
    df.to_csv(os.path.join(SALIDA, "fotocasa_andalucia_limpio.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(informe).to_csv(os.path.join(SALIDA, "informe_limpieza.csv"), index=False, encoding="utf-8-sig")

    print(f"\nSalida: {len(df):,} anuncios x {df.shape[1]} columnas")
    print("Por provincia:", df["provincia"].value_counts().to_dict())
    print("Por tipo:", df["tipo_vivienda"].value_counts().to_dict())
    nulos = (df.isna().mean() * 100).round(1)
    print("Columnas con nulos (%):", nulos[nulos > 0].sort_values(ascending=False).to_dict())


if __name__ == "__main__":
    main()
