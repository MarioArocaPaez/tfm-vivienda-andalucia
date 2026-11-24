"""
Representación del texto de los anuncios con un modelo de lenguaje (embeddings).

Las 14 palabras clave de variables_extra.py solo capturan una parte de lo que dice
la descripción. Aquí cada anuncio (título + descripción) se convierte en un vector
de 384 números con un modelo de lenguaje preentrenado multilingüe
(intfloat/multilingual-e5-small, ~470 MB). Textos con significado parecido
("reformado con calidades de lujo", "reforma integral de alta gama") quedan cerca
en ese espacio aunque no compartan palabras.

El cálculo se hace una sola vez y se guarda; el modelo de precios usa después ese
fichero, sin necesidad de volver a cargar el modelo de lenguaje.

Requiere (solo para este paso):
    pip install sentence-transformers

Uso, desde la raíz del repositorio (la primera vez descarga el modelo):
    python src/embeddings_texto.py

Salida:
    data/procesado/embeddings_descripcion.npz   (id_anuncio + matriz n x 384)
"""

import os
import time

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, "data", "procesado", "fotocasa_andalucia_limpio.csv")
SALIDA = os.path.join(RAIZ, "data", "procesado", "embeddings_descripcion.npz")
MODELO = "intfloat/multilingual-e5-small"
LONGITUD_MAXIMA = 512  # tokens; cubre la gran mayoría de descripciones


def main():
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(ENTRADA, usecols=["id_anuncio", "titulo", "descripcion"], dtype={"id_anuncio": str})
    # Los modelos E5 esperan el prefijo "passage: " en los textos a representar
    textos = ("passage: " + df["titulo"].fillna("") + ". " + df["descripcion"].fillna("")).tolist()

    print(f"Cargando {MODELO}...")
    modelo = SentenceTransformer(MODELO, device="cpu")
    modelo.max_seq_length = LONGITUD_MAXIMA

    print(f"Calculando embeddings de {len(textos)} anuncios (puede tardar varios minutos)...")
    inicio = time.time()
    # Ordenar por longitud acelera mucho el cálculo por lotes en CPU
    orden = np.argsort([len(t) for t in textos])
    vectores = modelo.encode([textos[i] for i in orden], batch_size=32, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)
    emb = np.empty_like(vectores)
    emb[orden] = vectores

    np.savez_compressed(SALIDA, id_anuncio=df["id_anuncio"].to_numpy(), embeddings=emb.astype(np.float32),
                        modelo=np.array(MODELO))
    print(f"Guardado {SALIDA}: {emb.shape} en {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
