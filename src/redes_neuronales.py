"""
Experimentos con redes neuronales sobre el modelo ampliado (texto + geo + renta).

El modelo tabular de referencia es el mejor modelo ampliado según la validación
cruzada (resultados_modelos.csv del reentreno con --grupos texto geo renta).

1. MLP (perceptrón multicapa) con las mismas variables tabulares.
   - Variables asimétricas en log1p, imputación por mediana con indicadores,
     estandarización de las variables y también del objetivo (sin esto la red no
     converge a un resultado útil).
   - Configuración elegida con una búsqueda manual breve en validación cruzada.
   - Se evalúa sola y combinada con el modelo tabular (media ponderada de log(precio)).

2. Corrección por texto con un modelo de lenguaje.
   - Cada anuncio se representa con el embedding de su título y descripción
     (src/embeddings_texto.py, multilingual-e5-small, 384 dimensiones).
   - Modelo en dos etapas: el modelo tabular predice log(precio); una regresión
     ridge sobre el embedding predice el error que comete (residuo fuera de
     muestra) y se suma a su predicción.
   - El residuo con el que se entrena la ridge se obtiene siempre fuera de muestra
     (validación cruzada interna), para que la ridge aprenda el error real del
     modelo y no el error, mucho menor, que tiene sobre sus propios datos de ajuste.

Ambos se evalúan con las mismas 5 particiones de validación cruzada que el resto de
modelos (la corrección por texto, anidada dentro de cada partición) y en el
conjunto de prueba.

Uso (requiere data/procesado/embeddings_descripcion.npz y el reentreno ampliado):
    python src/redes_neuronales.py

Salida en data/procesado/resultados/texto_geo_renta/:
    redes_neuronales.csv               métricas de cada variante
    predicciones_correccion_texto.csv  predicciones en prueba del modelo corregido
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelado import CV, RESULTADOS, construir, metricas, metricas_cv  # noqa: E402
from preprocesado import cargar_datos, preparar  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

GRUPOS = ("texto", "geo", "renta")
SALIDA = os.path.join(RESULTADOS, "_".join(GRUPOS))
EMBEDDINGS = os.path.join(os.path.dirname(RESULTADOS), "embeddings_descripcion.npz")

ASIMETRICAS = ["superficie_m2", "superficie_terreno_m2", "dist_costa_km", "dist_capital_km",
               "mivau_trans_total_4t", "mivau_trans_por_1000hab"]
MLP_PARAMS = dict(hidden_layer_sizes=(256, 128, 64), alpha=3.0, learning_rate_init=3e-4)
SEMILLAS_MLP = (0, 1, 2)
ALPHAS_RIDGE = np.logspace(-2, 3, 12)
INTERNA = KFold(5, shuffle=True, random_state=7)


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------
def construir_mlp(columnas, semilla):
    asim = [c for c in ASIMETRICAS if c in columnas]
    red = make_pipeline(
        ColumnTransformer([("log", FunctionTransformer(np.log1p), asim)], remainder="passthrough"),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        MLPRegressor(early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, max_iter=1000,
                     random_state=semilla, **MLP_PARAMS),
    )
    return TransformedTargetRegressor(regressor=red, transformer=StandardScaler())


def predecir_mlp(X_a, y_a, X_b):
    """Media de varias semillas (reduce la variabilidad propia de la red)."""
    return np.mean([construir_mlp(X_a.columns, s).fit(X_a, y_a).predict(X_b) for s in SEMILLAS_MLP], axis=0)


# ----------------------------------------------------------------------
# Corrección por texto
# ----------------------------------------------------------------------
class CorreccionTexto:
    """Modelo tabular + ridge sobre el embedding que predice su residuo."""

    def __init__(self, nombre, params):
        self.nombre, self.params = nombre, params

    def fit(self, X, y, E):
        residuo_oof = y.values - cross_val_predict(construir(self.nombre, self.params), X, y, cv=INTERNA)
        self.base_ = construir(self.nombre, self.params).fit(X, y)
        self.ridge_ = RidgeCV(alphas=ALPHAS_RIDGE).fit(E, residuo_oof)
        return self

    def predict(self, X, E, con_texto=True):
        p = self.base_.predict(X)
        return p + self.ridge_.predict(E) if con_texto else p


# ----------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cargar_embeddings(ids):
    z = np.load(EMBEDDINGS, allow_pickle=True)
    E = pd.DataFrame(z["embeddings"], index=z["id_anuncio"].astype(str))
    return E.loc[ids].to_numpy()


def main():
    X_train, X_test, y_train, y_test, ctx = preparar(grupos=GRUPOS)
    particion = pd.read_csv(os.path.join(os.path.dirname(RESULTADOS), "particion.csv"), dtype={"id_anuncio": str})
    ids = cargar_datos().merge(particion, on="id_anuncio")
    E_train = cargar_embeddings(ids.loc[ids["conjunto"] == "entrenamiento", "id_anuncio"])
    E_test = cargar_embeddings(ids.loc[ids["conjunto"] == "prueba", "id_anuncio"])
    assert len(E_train) == len(X_train) and len(E_test) == len(X_test)
    res = pd.read_csv(os.path.join(SALIDA, "resultados_modelos.csv"))
    tabular = res[res["modelo"] != "lineal"].sort_values("CV_RMSE_log").iloc[0]["modelo"]
    with open(os.path.join(SALIDA, "hiperparametros.json"), encoding="utf-8") as f:
        params = json.load(f)[tabular]
    print(f"Modelo tabular de referencia: {tabular}")

    # --- validación cruzada (mismas particiones que el resto del trabajo) ---
    t0 = time.time()
    cv = {k: [] for k in ["tabular", "mlp", "mezcla", "correccion_texto"]}
    oof_l, oof_m, oof_c = np.zeros(len(y_train)), np.zeros(len(y_train)), np.zeros(len(y_train))
    for i, (a, b) in enumerate(CV.split(X_train)):
        Xa, Xb, ya, yb = X_train.iloc[a], X_train.iloc[b], y_train.iloc[a], y_train.iloc[b]
        oof_m[b] = predecir_mlp(Xa, ya, Xb)
        corr = CorreccionTexto(tabular, params).fit(Xa, ya, E_train[a])
        oof_l[b] = corr.predict(Xb, E_train[b], con_texto=False)
        cv["tabular"].append(rmse(yb, oof_l[b]))
        cv["mlp"].append(rmse(yb, oof_m[b]))
        oof_c[b] = corr.predict(Xb, E_train[b])
        cv["correccion_texto"].append(rmse(yb, oof_c[b]))
        print(f"  partición {i + 1}/5 ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # peso de la mezcla modelo tabular + MLP elegido con las predicciones fuera de muestra
    pesos = np.round(np.arange(0, 0.51, 0.05), 2)
    peso = float(pesos[np.argmin([rmse(y_train, (1 - w) * oof_l + w * oof_m) for w in pesos])])
    for a, b in CV.split(X_train):
        cv["mezcla"].append(rmse(y_train.iloc[b], (1 - peso) * oof_l[b] + peso * oof_m[b]))
    corr_residuos = float(np.corrcoef(y_train - oof_l, y_train - oof_m)[0, 1])

    # --- conjunto de prueba ---
    corr = CorreccionTexto(tabular, params).fit(X_train, y_train, E_train)
    pred = {"tabular": corr.predict(X_test, E_test, con_texto=False),
            "mlp": predecir_mlp(X_train, y_train, X_test)}
    pred["mezcla"] = (1 - peso) * pred["tabular"] + peso * pred["mlp"]
    pred["correccion_texto"] = corr.predict(X_test, E_test)

    oof = {"tabular": oof_l, "mlp": oof_m, "mezcla": (1 - peso) * oof_l + peso * oof_m, "correccion_texto": oof_c}
    etiqueta = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
                "random_forest": "Random Forest"}[tabular]
    nombres = {"tabular": f"{etiqueta} ampliado", "mlp": "MLP", "mezcla": f"{etiqueta} + MLP (peso MLP {peso:.2f})",
               "correccion_texto": f"{etiqueta} ampliado + corrección por texto"}
    filas = []
    for k, p in pred.items():
        m = metricas(y_test.values, p)
        filas.append(dict(variante=k, descripcion=nombres[k], CV_RMSE_log=np.mean(cv[k]),
                          CV_RMSE_log_desv=np.std(cv[k]), **{f"CV_RMSE_log_p{i + 1}": v for i, v in enumerate(cv[k])},
                          **metricas_cv(y_train, oof[k]),
                          **{f"CV_MAE_eur_p{i + 1}": metricas(y_train.iloc[te], oof[k][te])["MAE_eur"]
                             for i, (_, te) in enumerate(CV.split(X_train))},
                          **{f"test_{n}": v for n, v in m.items()}))
        mc = metricas_cv(y_train, oof[k])
        print(f"{nombres[k]:45s} CV: MAE {mc['CV_MAE_eur']:,.0f} € R² {mc['CV_R2_eur']:.3f} "
              f"(RMSE log {np.mean(cv[k]):.4f}) | prueba: MAE {m['MAE_eur']:,.0f} € MdAPE {m['MdAPE_pct']:.1f} % "
              f"R² {m['R2_eur']:.3f}")
    print(f"Correlación de residuos {etiqueta}-MLP: {corr_residuos:.2f} | alpha ridge: {corr.ridge_.alpha_:.2f}")

    res = pd.DataFrame(filas)
    res["modelo_tabular"] = tabular
    res["corr_residuos_tabular_mlp"] = corr_residuos
    res["alpha_ridge"] = corr.ridge_.alpha_
    res.to_csv(os.path.join(SALIDA, "redes_neuronales.csv"), index=False, encoding="utf-8-sig")
    salida_pred = ctx[["id_anuncio", "provincia", "tipo_vivienda", "es_capital", "precio_eur"]].copy()
    for k in ("tabular", "correccion_texto"):
        salida_pred[f"pred_{k}"] = np.exp(pred[k])
    salida_pred.to_csv(os.path.join(SALIDA, "predicciones_correccion_texto.csv"), index=False, encoding="utf-8-sig")
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
