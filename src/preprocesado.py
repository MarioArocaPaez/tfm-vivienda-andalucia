"""
Preprocesado común a todos los modelos.

Define qué variables entran en el modelo, cómo se codifican y la partición
entrenamiento/prueba. Lo importan src/modelado.py y el notebook de modelado,
de modo que todos los modelos usan exactamente los mismos datos.

Decisiones:
- Variable objetivo: log(precio). Las métricas se calculan en euros tras
  deshacer la transformación.
- Partición 80/20 estratificada por provincia, con semilla fija; se guarda en
  data/procesado/particion.csv para que sea reproducible y auditable.
- Quedan fuera las variables derivadas del precio (precio_m2, bajada_precio_eur),
  los identificadores y los textos libres.
- Las variables que describen el ANUNCIO y no la vivienda (nº de fotos, longitud
  de la descripción, bajada de precio) no entran en el modelo principal; se
  evalúan aparte (ablación), ya que dependen de cómo se anuncia y no de qué se vende.

Uso:
    from preprocesado import cargar_datos, preparar, VARIABLES
    python src/preprocesado.py      -> genera la partición y muestra un resumen
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import variables_extra

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESADO = os.path.join(RAIZ, "data", "procesado")
DATOS = os.path.join(PROCESADO, "fotocasa_andalucia_limpio.csv")
PARTICION = os.path.join(PROCESADO, "particion.csv")

SEMILLA = 42
PROPORCION_TEST = 0.20
OBJETIVO = "log_precio"

# ----------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------
NUMERICAS = [
    "superficie_m2", "log_superficie", "habitaciones", "banos", "planta_num",
    "superficie_terreno_m2", "antiguedad_anos",
    "latitud", "longitud",
    "log_poblacion", "mivau_trans_por_1000hab", "mivau_trans_total_4t", "mivau_valor_medio_ref_eur",
]
BINARIAS = [
    "ascensor", "terraza", "balcon", "parking", "piscina", "jardin", "aire_acondicionado",
    "calefaccion", "trastero", "cocina_equipada", "patio", "puerta_blindada", "electrodomesticos",
    "domotica", "obra_nueva", "amueblado", "alquilado", "es_capital",
]
CATEGORICAS = ["tipo_vivienda", "estado", "orientacion", "provincia"]

# Variables del anuncio: solo para la ablación.
VARIABLES_ANUNCIO = ["num_fotos", "longitud_descripcion", "tiene_bajada_precio"]

VARIABLES = NUMERICAS + BINARIAS + CATEGORICAS

# Para describir resultados (no entran en el modelo)
COLUMNAS_CONTEXTO = ["id_anuncio", "provincia", "municipio_ine", "es_capital", "tipo_vivienda",
                     "precio_eur", "superficie_m2", "zona"]


def cargar_datos(ruta=DATOS):
    df = pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)
    df["log_superficie"] = np.log(df["superficie_m2"])
    df["log_poblacion"] = np.log(df["poblacion_2025"])
    for c in BINARIAS + ["tiene_bajada_precio"]:
        df[c] = df[c].astype(str).str.lower().eq("true").astype(int)
    for c in CATEGORICAS:
        df[c] = df[c].astype(str)
    df["id_anuncio"] = df["id_anuncio"].astype(str)
    return df


def generar_particion(df):
    entrenamiento, prueba = train_test_split(
        df["id_anuncio"], test_size=PROPORCION_TEST, random_state=SEMILLA, stratify=df["provincia"])
    particion = pd.concat([pd.DataFrame({"id_anuncio": entrenamiento, "conjunto": "entrenamiento"}),
                           pd.DataFrame({"id_anuncio": prueba, "conjunto": "prueba"})])
    particion.to_csv(PARTICION, index=False, encoding="utf-8-sig")
    return particion


def preparar(con_anuncio=False, regenerar_particion=False, grupos=()):
    """Devuelve X_train, X_test, y_train, y_test y el DataFrame completo de prueba.

    grupos: grupos de variables adicionales de variables_extra ("texto", "geo", "renta").

    X conserva los ausentes (NaN): cada modelo decide cómo tratarlos.
    Las categóricas se codifican one-hot con las categorías vistas en entrenamiento.
    """
    df = cargar_datos()
    if regenerar_particion or not os.path.exists(PARTICION):
        particion = generar_particion(df)
    else:
        particion = pd.read_csv(PARTICION, dtype={"id_anuncio": str})
    df = df.merge(particion, on="id_anuncio", how="inner")
    df = variables_extra.anadir(df, grupos)

    variables = VARIABLES + (VARIABLES_ANUNCIO if con_anuncio else [])
    variables += [v for g in grupos for v in variables_extra.GRUPOS[g]]
    X = pd.get_dummies(df[variables], columns=CATEGORICAS, dtype=int)
    entreno = (df["conjunto"] == "entrenamiento").values

    X_train, X_test = X[entreno].reset_index(drop=True), X[~entreno].reset_index(drop=True)
    # Columnas one-hot sin ningún caso en entrenamiento no aportan nada
    constantes = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    X_train, X_test = X_train.drop(columns=constantes), X_test.drop(columns=constantes)

    y_train = df.loc[entreno, OBJETIVO].reset_index(drop=True)
    y_test = df.loc[~entreno, OBJETIVO].reset_index(drop=True)
    contexto_test = df.loc[~entreno].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, contexto_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, ctx = preparar(regenerar_particion=True)
    print(f"Entrenamiento: {len(X_train):,} | prueba: {len(X_test):,} | variables: {X_train.shape[1]}")
    print("Ausentes en entrenamiento (%):",
          (X_train.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())
    df = cargar_datos().merge(pd.read_csv(PARTICION, dtype={"id_anuncio": str}), on="id_anuncio")
    print((pd.crosstab(df["provincia"], df["conjunto"], normalize="columns") * 100).round(1))
