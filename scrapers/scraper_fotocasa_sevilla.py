"""Lanza el scraper de listados para la provincia de Sevilla."""

from fotocasa_base import ejecutar_provincia
from provincias import PROVINCIAS


if __name__ == "__main__":
    config = PROVINCIAS["sevilla"]
    ejecutar_provincia(
        provincia=config["nombre"],
        url_provincia=config["url"],
    )

