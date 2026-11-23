"""
Ablación de los grupos de variables adicionales (src/variables_extra.py).

Toma el mejor modelo del reentreno con todas las variables
(data/procesado/resultados/<grupos>/) y, con sus hiperparámetros fijos, lo evalúa
añadiendo y quitando cada grupo:

    base                      solo las variables del modelo original
    base + <grupo>            cada grupo por separado
    todas                     todos los grupos
    todas - <grupo>           todos menos uno (aportación marginal de cada grupo)

Para cada variante: RMSE de log(precio) en validación cruzada (mismas 5 particiones),
métricas en prueba y sesgo mediano por quintil de precio real en prueba (el problema
de los extremos que motivó estas variables).

Uso (desde la raíz del repositorio, tras `python src/modelado.py --grupos texto geo renta`):
    python src/ablacion_variables.py --grupos texto geo renta

Salida:
    data/procesado/resultados/ablacion_variables.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelado import RESULTADOS, construir, metricas, rmse_cv  # noqa: E402
from preprocesado import preparar  # noqa: E402


def sesgo_quintiles(y_log, pred_log):
    """Error relativo mediano (predicho - real) / real por quintil de precio real."""
    y, p = pd.Series(np.exp(y_log)), pd.Series(np.exp(pred_log))
    q = pd.qcut(y, 5, labels=[f"Q{i}" for i in range(1, 6)])
    return ((p - y) / y).groupby(q, observed=True).median().mul(100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupos", nargs="+", default=["texto", "geo", "renta"])
    args = ap.parse_args()
    todos = tuple(args.grupos)

    origen = os.path.join(RESULTADOS, "_".join(todos))
    res = pd.read_csv(os.path.join(origen, "resultados_modelos.csv"))
    mejor = res.sort_values("CV_RMSE_log").iloc[0]["modelo"]
    with open(os.path.join(origen, "hiperparametros.json"), encoding="utf-8") as f:
        params = json.load(f)[mejor]
    print(f"Modelo: {mejor} (hiperparámetros de {origen})\n")

    variantes = {"base": ()}
    variantes.update({f"base + {g}": (g,) for g in todos})
    variantes["todas"] = todos
    if len(todos) > 2:
        variantes.update({f"todas - {g}": tuple(x for x in todos if x != g) for g in todos})

    filas = []
    for nombre, grupos in variantes.items():
        X_train, X_test, y_train, y_test, _ = preparar(grupos=grupos)
        cv_media, cv_desv = rmse_cv(construir(mejor, params), X_train, y_train)
        pred = construir(mejor, params).fit(X_train, y_train).predict(X_test)
        m = metricas(y_test.values, pred)
        sq = sesgo_quintiles(y_test.values, pred)
        filas.append(dict(modelo=mejor, variante=nombre, grupos="+".join(grupos), n_variables=X_train.shape[1],
                          CV_RMSE_log=cv_media, CV_RMSE_log_desv=cv_desv,
                          **{f"test_{k}": v for k, v in m.items()},
                          **{f"sesgo_{k}_pct": v for k, v in sq.items()}))
        print(f"{nombre:18s} CV {cv_media:.4f} ± {cv_desv:.4f} | prueba MdAPE {m['MdAPE_pct']:.1f} % "
              f"MAE {m['MAE_eur']:,.0f} € | sesgo Q1 {sq['Q1']:+.1f} % Q5 {sq['Q5']:+.1f} %", flush=True)

    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(RESULTADOS, "ablacion_variables.csv"), index=False, encoding="utf-8-sig")
    print(f"\nGuardado en {os.path.join(RESULTADOS, 'ablacion_variables.csv')}")


if __name__ == "__main__":
    main()
