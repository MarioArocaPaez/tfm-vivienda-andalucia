"""Lanza el scraper de listados para la provincia de Almería."""

from fotocasa_base import ejecutar_provincia
from provincias import PROVINCIAS


if __name__ == "__main__":
    config = PROVINCIAS["almeria"]
    ejecutar_provincia(
        provincia=config["nombre"],
        url_provincia=config["url"],
    )