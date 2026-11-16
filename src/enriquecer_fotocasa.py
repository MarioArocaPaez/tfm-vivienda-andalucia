"""
Enriquece el dataset de Fotocasa con variables municipales (INE) y de mercado (MIVAU).

El cruce se hace por código INE del municipio: los nombres de Fotocasa, INE y
MIVAU no coinciden siempre (artículos, tildes, "Capital", zonas tratadas como
municipio...), así que primero se normalizan y se resuelven a su código oficial.

Uso (desde la raíz del repositorio):
    python src/enriquecer_fotocasa.py [fotocasa.csv] [carpeta_ine] [carpeta_mivau_procesado] [carpeta_salida]
Por defecto:
    data/procesado/fotocasa_andalucia_combinado.csv   (salida de combinar_fotocasa.py)
    data/externos/ine                                 (xlsx "Población por municipios y sexo", una por provincia)
    data/procesado                                    (salida de procesar_mivau.py)
    data/procesado

Variables añadidas:
    cod_ine, municipio_ine        municipio oficial (INE) del anuncio
    poblacion_2025                padrón a 1/1/2025 (INE)
    mivau_trans_segunda_4t / mivau_trans_nueva_4t / mivau_trans_total_4t
                                  compraventas del municipio en los 4 últimos trimestres publicados
    mivau_trans_por_1000hab       compraventas totales / población * 1000
    mivau_valor_medio_segunda_eur / mivau_valor_medio_nueva_eur
                                  valor medio provincial de compraventa de vivienda libre
    mivau_ref_tipo, mivau_valor_medio_ref_eur
                                  referencia aplicable al anuncio: nueva si es obra nueva
                                  o tiene menos de 1 año; segunda mano en el resto

Salida:
    fotocasa_andalucia_enriquecido.csv
    correspondencia_municipios_fotocasa_ine.csv   (para documentar el cruce)
"""

import glob
import os
import re
import sys
import unicodedata

import openpyxl
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESADO = os.path.join(RAIZ, "data", "procesado")

args = sys.argv[1:] + [None] * 4
FOTOCASA_CSV = args[0] or os.path.join(PROCESADO, "fotocasa_andalucia_combinado.csv")
INE_DIR = args[1] or os.path.join(RAIZ, "data", "externos", "ine")
MIVAU_DIR = args[2] or PROCESADO
SALIDA = args[3] or PROCESADO

PROV_COD = {"04": "Almería", "11": "Cádiz", "14": "Córdoba", "18": "Granada",
            "21": "Huelva", "23": "Jaén", "29": "Málaga", "41": "Sevilla"}

# Zonas o urbanizaciones que Fotocasa trata como municipio -> municipio INE.
ZONAS = {
    ("Cádiz", "Sotogrande"): "San Roque",
    ("Cádiz", "La Alcaidesa"): "San Roque",
    ("Huelva", "El Portil"): "Punta Umbría",
    ("Huelva", "El Rompido"): "Cartaya",
    ("Huelva", "La Antilla"): "Lepe",
    ("Huelva", "Islantilla"): "Lepe",          # compartida con Isla Cristina
    ("Sevilla", "Montequinto"): "Dos Hermanas",
}
# Nombres del MIVAU que no coinciden con el nombre oficial del INE.
ALIAS_MIVAU = {"Otura": "Villa de Otura", "Torre del Campo": "Torredelcampo",
               "Lantejuela (La)": "Lantejuela"}
# Pseudo-municipio que aparece en la navegación de Fotocasa.
NO_MUNICIPIO = {"pisos de segunda mano"}


def norm(texto):
    texto = str(texto).strip()
    texto = re.sub(r"\s+capital$", "", texto, flags=re.I)
    m = re.match(r"^(.*?),\s*(el|la|los|las|l')$", texto, flags=re.I)   # INE:   "Cuervo de Sevilla, El"
    if m:
        texto = f"{m.group(2)} {m.group(1)}"
    m = re.match(r"^(.*?)\s*\((el|la|los|las)\)$", texto, flags=re.I)   # MIVAU: "Cuervo de Sevilla (El)"
    if m:
        texto = f"{m.group(2)} {m.group(1)}"
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def cargar_ine():
    filas = []
    for ruta in glob.glob(os.path.join(INE_DIR, "*.xlsx")):
        hoja = openpyxl.load_workbook(ruta, read_only=True).worksheets[0]
        for fila in hoja.iter_rows(values_only=True):
            m = re.match(r"^(\d{5}) (.+)$", str(fila[0] or ""))
            if m and fila[1] is not None:
                filas.append(dict(cod_ine=m.group(1), municipio_ine=m.group(2).strip(),
                                  provincia=PROV_COD.get(m.group(1)[:2]),
                                  poblacion_2025=int(fila[1])))
    ine = pd.DataFrame(filas).drop_duplicates("cod_ine")
    ine["clave"] = ine.municipio_ine.map(norm)
    return ine


def cargar_mivau_municipal(ine):
    mm = pd.read_csv(os.path.join(MIVAU_DIR, "mivau_transacciones_municipios_andalucia.csv"))
    mm["periodo"] = mm.anio * 10 + mm.trimestre
    ultimos = sorted(mm.periodo.unique())[-4:]
    agregado = (mm[mm.periodo.isin(ultimos)]
                .groupby(["provincia", "municipio", "tipo"]).n_transacciones.sum()
                .unstack("tipo").reset_index())
    agregado["clave"] = agregado.municipio.replace(ALIAS_MIVAU).map(norm)
    miv = agregado.merge(ine[["provincia", "clave", "cod_ine"]], on=["provincia", "clave"], how="left")
    sin_codigo = miv[miv.cod_ine.isna()].municipio.tolist()
    miv = (miv.dropna(subset=["cod_ine"])
              .rename(columns={"segunda_mano": "mivau_trans_segunda_4t",
                               "nueva": "mivau_trans_nueva_4t"}))
    return miv[["cod_ine", "mivau_trans_segunda_4t", "mivau_trans_nueva_4t"]], ultimos, sin_codigo


def cargar_mivau_provincial(ultimos):
    mp = pd.read_csv(os.path.join(MIVAU_DIR, "mivau_valor_provincias_andalucia.csv"))
    mp = mp[(mp.anio * 10 + mp.trimestre).isin(ultimos) & mp.metrica.str.startswith("valor_medio")]
    return (mp.pivot_table(index="ambito", columns="metrica", values="valor", aggfunc="mean")
              .rename(columns={"valor_medio_libre_segunda_mano_eur": "mivau_valor_medio_segunda_eur",
                               "valor_medio_libre_nueva_eur": "mivau_valor_medio_nueva_eur"})
              .reset_index().rename(columns={"ambito": "provincia"}))


def main():
    os.makedirs(SALIDA, exist_ok=True)
    ine = cargar_ine()
    miv, ultimos, sin_codigo = cargar_mivau_municipal(ine)
    ref = cargar_mivau_provincial(ultimos)

    df = pd.read_csv(FOTOCASA_CSV, encoding="utf-8-sig", low_memory=False)

    # Correspondencia municipio Fotocasa -> municipio INE
    mun = df[["provincia", "municipio"]].drop_duplicates().copy()
    mun["municipio_buscar"] = [ZONAS.get((p, m), m) for p, m in zip(mun.provincia, mun.municipio)]
    mun["clave"] = mun.municipio_buscar.map(norm)
    mun = mun.merge(ine[["provincia", "clave", "cod_ine", "municipio_ine", "poblacion_2025"]],
                    on=["provincia", "clave"], how="left")
    mun.loc[mun.municipio.map(norm).isin(NO_MUNICIPIO), ["cod_ine", "municipio_ine", "poblacion_2025"]] = None
    mun["es_zona"] = [(p, m) in ZONAS for p, m in zip(mun.provincia, mun.municipio)]
    mun.drop(columns=["clave", "municipio_buscar"]).to_csv(
        os.path.join(SALIDA, "correspondencia_municipios_fotocasa_ine.csv"),
        index=False, encoding="utf-8-sig")

    out = (df.merge(mun[["provincia", "municipio", "cod_ine", "municipio_ine", "poblacion_2025"]],
                    on=["provincia", "municipio"], how="left")
             .merge(miv, on="cod_ine", how="left")
             .merge(ref, on="provincia", how="left"))
    out["mivau_trans_total_4t"] = out[["mivau_trans_segunda_4t", "mivau_trans_nueva_4t"]].sum(axis=1, min_count=1)
    out["mivau_trans_por_1000hab"] = (out.mivau_trans_total_4t / out.poblacion_2025 * 1000).round(2)
    es_nueva = (out.obra_nueva.astype(str).str.lower() == "true") | \
               (out.antiguedad.astype(str).str.strip() == "Menos de 1 año")
    out["mivau_ref_tipo"] = es_nueva.map({True: "nueva", False: "segunda_mano"})
    out["mivau_valor_medio_ref_eur"] = out.mivau_valor_medio_segunda_eur.where(
        ~es_nueva, out.mivau_valor_medio_nueva_eur).round(0)
    out.to_csv(os.path.join(SALIDA, "fotocasa_andalucia_enriquecido.csv"), index=False, encoding="utf-8-sig")

    sin_ine = mun[mun.cod_ine.isna()].municipio.unique().tolist()
    print(f"INE: {len(ine)} municipios | MIVAU sin código INE: {sin_codigo}")
    print(f"Trimestres MIVAU usados: {[int(p) for p in ultimos]}")
    print(f"Fotocasa: {len(mun)} municipios | sin código INE: {sin_ine}")
    print(f"Anuncios: {len(out):,} | con código INE {out.cod_ine.notna().mean():.1%} | "
          f"con transacciones {out.mivau_trans_total_4t.notna().mean():.1%} | referencia nueva: {es_nueva.sum()}")


if __name__ == "__main__":
    main()
