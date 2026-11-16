"""Ejecuta secuencialmente el scraper de listados para las ocho provincias."""

import time

from fotocasa_base import ejecutar_provincia
from provincias import PROVINCIAS


if __name__ == "__main__":
    for indice, config in enumerate(PROVINCIAS.values(), start=1):
        print("\n" + "=" * 70)
        print(f"Provincia {indice}/{len(PROVINCIAS)}: {config['nombre']}")
        print("=" * 70)

        try:
            ejecutar_provincia(
                provincia=config["nombre"],
                url_provincia=config["url"],
            )
        except Exception as exc:
            print(f"Error en {config['nombre']}: {exc}")

        if indice < len(PROVINCIAS):
            time.sleep(10)
