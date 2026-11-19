"""
Entrenamiento, ajuste y evaluación de los modelos de predicción de precio.

Modelos:
    - Regresión lineal (referencia), con imputación por mediana e indicadores de ausencia.
    - Random Forest, XGBoost, LightGBM y CatBoost, con ausentes tratados de forma nativa.

Procedimiento:
    1. Ajuste de hiperparámetros con Optuna (muestreador TPE, semilla fija), minimizando
       el RMSE de log(precio) en validación cruzada de 5 particiones sobre ENTRENAMIENTO.
       El conjunto de prueba no se usa en ningún momento del ajuste.
    2. Reentrenamiento con los mejores hiperparámetros sobre todo el entrenamiento.
    3. Evaluación única en PRUEBA, en euros (tras deshacer el logaritmo).
    4. Ablación: el mejor modelo, con y sin las variables del anuncio
       (nº de fotos, longitud de la descripción, bajada de precio), con las mismas particiones.

Uso (desde la raíz del repositorio):
    python src/modelado.py              # 30 pruebas de Optuna por modelo
    python src/modelado.py --pruebas 60
    python src/modelado.py --modelos lightgbm catboost

Salida:
    data/procesado/resultados/resultados_modelos.csv     métricas CV y prueba por modelo
    data/procesado/resultados/predicciones_prueba.csv    predicción de cada modelo por vivienda
    data/procesado/resultados/hiperparametros.json       mejores hiperparámetros
    data/procesado/resultados/historial_optuna.csv       todas las pruebas de Optuna
    data/procesado/resultados/ablacion_anuncio.csv       efecto de las variables del anuncio
    models/<modelo>.joblib                               modelos entrenados
"""

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocesado import RAIZ, SEMILLA, preparar  # noqa: E402

RESULTADOS = os.path.join(RAIZ, "data", "procesado", "resultados")
MODELOS_DIR = os.path.join(RAIZ, "models")
N_JOBS = max(1, (os.cpu_count() or 2))
CV = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------

def metricas(y_log, pred_log):
    """Métricas en euros (y R² también en escala logarítmica)."""
    y, p = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(p - y) / y
    return {
        "MAE_eur": mean_absolute_error(y, p),
        "RMSE_eur": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE_pct": float(ape.mean() * 100),
        "MdAPE_pct": float(np.median(ape) * 100),
        "R2_eur": r2_score(y, p),
        "R2_log": r2_score(y_log, pred_log),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_log, pred_log))),
    }


# ----------------------------------------------------------------------
# Espacios de búsqueda
# ----------------------------------------------------------------------

def construir(nombre, params):
    if nombre == "lineal":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             StandardScaler(), LinearRegression())
    if nombre == "random_forest":
        return RandomForestRegressor(random_state=SEMILLA, n_jobs=N_JOBS, **params)
    if nombre == "xgboost":
        return XGBRegressor(random_state=SEMILLA, n_jobs=N_JOBS, tree_method="hist", **params)
    if nombre == "lightgbm":
        return LGBMRegressor(random_state=SEMILLA, n_jobs=N_JOBS, verbose=-1, **params)
    if nombre == "catboost":
        return CatBoostRegressor(random_seed=SEMILLA, thread_count=N_JOBS, verbose=0,
                                 allow_writing_files=False, **params)
    raise ValueError(nombre)


def espacio(nombre, t):
    if nombre == "random_forest":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 700, step=100),
            max_features=t.suggest_float("max_features", 0.2, 1.0),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
            max_depth=t.suggest_categorical("max_depth", [None, 15, 25, 40]),
        )
    if nombre == "xgboost":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 1500, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=t.suggest_int("max_depth", 3, 10),
            min_child_weight=t.suggest_float("min_child_weight", 1, 20, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
    if nombre == "lightgbm":
        return dict(
            n_estimators=t.suggest_int("n_estimators", 200, 1500, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=t.suggest_int("num_leaves", 15, 127),
            min_child_samples=t.suggest_int("min_child_samples", 5, 80),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
    if nombre == "catboost":
        return dict(
            iterations=t.suggest_int("iterations", 300, 1200, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            depth=t.suggest_int("depth", 4, 8),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 20, log=True),
            bagging_temperature=t.suggest_float("bagging_temperature", 0, 1),
        )
    return {}


# ----------------------------------------------------------------------
# Ajuste y evaluación
# ----------------------------------------------------------------------

def rmse_cv(modelo, X, y):
    pred = cross_val_predict(modelo, X, y, cv=CV)
    por_particion = [np.sqrt(mean_squared_error(y.iloc[te], pred[te])) for _, te in CV.split(X)]
    return float(np.mean(por_particion)), float(np.std(por_particion))


def ajustar(nombre, X, y, pruebas, historial):
    if nombre == "lineal":
        return {}, *rmse_cv(construir(nombre, {}), X, y)

    def objetivo(t):
        return rmse_cv(construir(nombre, espacio(nombre, t)), X, y)[0]

    estudio = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    estudio.optimize(objetivo, n_trials=pruebas, show_progress_bar=False)
    h = estudio.trials_dataframe()
    h.insert(0, "modelo", nombre)
    historial.append(h)
    mejores = dict(estudio.best_params)
    if nombre == "lightgbm":
        mejores["subsample_freq"] = 1
    media, desv = rmse_cv(construir(nombre, mejores), X, y)
    return mejores, media, desv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruebas", type=int, default=30)
    ap.add_argument("--modelos", nargs="+",
                    default=["lineal", "random_forest", "xgboost", "lightgbm", "catboost"])
    args = ap.parse_args()

    os.makedirs(RESULTADOS, exist_ok=True)
    os.makedirs(MODELOS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, ctx = preparar()
    print(f"Entrenamiento {X_train.shape} | prueba {X_test.shape} | {args.pruebas} pruebas por modelo\n")

    filas, historial, hiper = [], [], {}
    predicciones = ctx[["id_anuncio", "provincia", "municipio_ine", "es_capital", "tipo_vivienda",
                        "precio_eur", "superficie_m2"]].copy()
    for nombre in args.modelos:
        t0 = time.time()
        mejores, cv_media, cv_desv = ajustar(nombre, X_train, y_train, args.pruebas, historial)
        modelo = construir(nombre, mejores).fit(X_train, y_train)
        pred = modelo.predict(X_test)
        m = metricas(y_test.values, pred)
        filas.append(dict(modelo=nombre, CV_RMSE_log=cv_media, CV_RMSE_log_desv=cv_desv,
                          **{f"test_{k}": v for k, v in m.items()}, minutos=(time.time() - t0) / 60))
        predicciones[f"pred_{nombre}"] = np.exp(pred)
        hiper[nombre] = mejores
        joblib.dump(modelo, os.path.join(MODELOS_DIR, f"{nombre}.joblib"))
        print(f"{nombre:14s} CV RMSE(log) {cv_media:.4f} ± {cv_desv:.4f} | prueba: MAPE {m['MAPE_pct']:.1f} % "
              f"MdAPE {m['MdAPE_pct']:.1f} % MAE {m['MAE_eur']:,.0f} € R²(log) {m['R2_log']:.3f} "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)

    res = pd.DataFrame(filas).sort_values("CV_RMSE_log")
    res.to_csv(os.path.join(RESULTADOS, "resultados_modelos.csv"), index=False, encoding="utf-8-sig")
    predicciones.to_csv(os.path.join(RESULTADOS, "predicciones_prueba.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RESULTADOS, "hiperparametros.json"), "w", encoding="utf-8") as f:
        json.dump(hiper, f, indent=2, ensure_ascii=False)
    if historial:
        pd.concat(historial).to_csv(os.path.join(RESULTADOS, "historial_optuna.csv"),
                                    index=False, encoding="utf-8-sig")

    # ---- Ablación: variables del anuncio, con el mejor modelo (elegido por CV) ----
    mejor = res.iloc[0]["modelo"]
    Xa_train, Xa_test, _, _, _ = preparar(con_anuncio=True)
    filas_ab = []
    for etiqueta, (Xtr, Xte) in {"sin variables del anuncio": (X_train, X_test),
                                 "con variables del anuncio": (Xa_train, Xa_test)}.items():
        cv_media, cv_desv = rmse_cv(construir(mejor, hiper[mejor]), Xtr, y_train)
        pred = construir(mejor, hiper[mejor]).fit(Xtr, y_train).predict(Xte)
        filas_ab.append(dict(modelo=mejor, variante=etiqueta, CV_RMSE_log=cv_media, CV_RMSE_log_desv=cv_desv,
                             **{f"test_{k}": v for k, v in metricas(y_test.values, pred).items()}))
    ab = pd.DataFrame(filas_ab)
    ab.to_csv(os.path.join(RESULTADOS, "ablacion_anuncio.csv"), index=False, encoding="utf-8-sig")

    print("\nMejor modelo (CV):", mejor)
    print(res[["modelo", "CV_RMSE_log", "test_MAPE_pct", "test_MdAPE_pct", "test_MAE_eur", "test_R2_log"]]
          .round(4).to_string(index=False))
    print("\nAblación:")
    print(ab[["variante", "CV_RMSE_log", "test_MAPE_pct", "test_MdAPE_pct", "test_R2_log"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
