"""
Une en un único CSV los anuncios con detalle de Fotocasa de las distintas extracciones.

Entradas (si existen):
    data/fotocasa_andalucia_detalle.csv           scraper_detalle_fotocasa.py (provincias)
    data/capitales/fotocasa_capitales_detalle.csv scraper_capitales.py

Si un anuncio aparece en ambas, se conserva una sola fila con este criterio:
    1. la que tiene el detalle validado;
    2. a igualdad, la extracción más reciente.

Uso (desde la raíz del repositorio):
    python src/combinar_fotocasa.py [csv1 csv2 ...]
Salida:
    data/procesado/fotocasa_andalucia_combinado.csv
"""

import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADAS = sys.argv[1:] or [
    os.path.join(RAIZ, "data", "fotocasa_andalucia_detalle.csv"),
    os.path.join(RAIZ, "data", "capitales", "fotocasa_capitales_detalle.csv"),
]
SALIDA = os.path.join(RAIZ, "data", "procesado", "fotocasa_andalucia_combinado.csv")


def main():
    partes = []
    for ruta in ENTRADAS:
        if not os.path.exists(ruta):
            print(f"No existe (se omite): {ruta}")
            continue
        df = pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)
        df["origen_extraccion"] = os.path.basename(ruta)
        partes.append(df)
        print(f"{os.path.basename(ruta)}: {len(df):,} anuncios")
    if not partes:
        sys.exit("No hay datos que combinar.")

    df = pd.concat(partes, ignore_index=True, sort=False)
    df["_validado"] = df["detalle_validado"].astype(str).str.lower() == "true"
    df["_fecha"] = pd.to_datetime(df["fecha_extraccion"], errors="coerce")
    df["_id"] = df["id_anuncio"].astype(str)
    antes = len(df)
    df = (df.sort_values(["_validado", "_fecha"], ascending=[False, False])
            .drop_duplicates("_id", keep="first")
            .drop(columns=["_validado", "_fecha", "_id"]))

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False, encoding="utf-8-sig")
    validados = (df["detalle_validado"].astype(str).str.lower() == "true").sum()
    print(f"Combinado: {len(df):,} anuncios únicos ({antes - len(df):,} duplicados eliminados) | "
          f"validados: {validados:,}")
    print(f"Salida: {SALIDA}")


if __name__ == "__main__":
    main()
