"""Lanza el scraper de listados para la provincia de Córdoba."""

from fotocasa_base import ejecutar_provincia
from provincias import PROVINCIAS


if __name__ == "__main__":
    config = PROVINCIAS["cordoba"]
    ejecutar_provincia(
        provincia=config["nombre"],
        url_provincia=config["url"],
    )