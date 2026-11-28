# Predicción de precios de vivienda en Andalucía mediante aprendizaje automático

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo Fin de Máster (TFM) del Máster Universitario en Ciencia de Datos de la Universitat Oberta de Catalunya (UOC).

El objetivo principal del proyecto es analizar y predecir el precio de oferta de viviendas en Andalucía mediante técnicas de análisis de datos y aprendizaje automático. Para ello, se combinan datos a nivel de inmueble procedentes de portales inmobiliarios con fuentes estadísticas oficiales que permiten incorporar información del contexto territorial, socioeconómico y del mercado de la vivienda.

El trabajo se plantea como un estudio reproducible de extremo a extremo: adquisición de datos, depuración, integración de fuentes, análisis exploratorio, ingeniería de variables, entrenamiento y evaluación de modelos, e interpretación de resultados.

## Objetivos

El objetivo general es estudiar hasta qué punto el precio de oferta de una vivienda en Andalucía puede explicarse y predecirse a partir de tres grandes grupos de variables:

1. Características propias del inmueble: superficie, número de habitaciones, baños, tipo de vivienda, planta, ascensor, terraza, piscina, aparcamiento, estado, antigüedad y otros atributos disponibles.
2. Localización: provincia, municipio, zona y, cuando sea posible, información geográfica derivada de coordenadas.
3. Contexto externo: variables socioeconómicas, demográficas, territoriales y de actividad del mercado inmobiliario procedentes de fuentes oficiales.

Entre las preguntas de investigación que orientan el trabajo se encuentran:

- ¿Qué variables tienen mayor capacidad explicativa sobre el precio de oferta de una vivienda?
- ¿Cuánto aporta la localización frente a las características físicas del inmueble?
- ¿Existen diferencias relevantes entre provincias, municipios y otras unidades territoriales?
- ¿Hasta qué punto mejora la predicción al incorporar variables socioeconómicas y de mercado?
- ¿Qué modelos ofrecen el mejor equilibrio entre capacidad predictiva e interpretabilidad?

## Ámbito geográfico

El estudio se centra en Andalucía y contempla sus ocho provincias:

- Almería
- Cádiz
- Córdoba
- Granada
- Huelva
- Jaén
- Málaga
- Sevilla

La intención es trabajar con el mayor nivel de granularidad territorial que resulte metodológicamente defendible y compatible con las distintas fuentes utilizadas.

## Fuentes de datos

### Fotocasa

Los anuncios inmobiliarios constituyen la fuente principal a nivel de inmueble. A partir de ellos se obtienen variables como:

- precio de oferta;
- superficie;
- precio por metro cuadrado;
- habitaciones y baños;
- tipo de vivienda;
- ubicación;
- planta;
- ascensor;
- terraza y balcón;
- garaje o aparcamiento;
- piscina;
- jardín;
- aire acondicionado y calefacción;
- trastero;
- estado y antigüedad, cuando están disponibles;
- descripción del anuncio;
- agencia o anunciante;
- coordenadas geográficas, cuando pueden obtenerse de forma fiable;
- información sobre disponibilidad del anuncio.

Es importante señalar que el precio observado en Fotocasa es un **precio de oferta**, no un precio efectivo de compraventa. Esta distinción se mantiene durante todo el análisis y condiciona la interpretación de los resultados.

### Ministerio de Vivienda y Agenda Urbana (MIVAU)

Se utilizan datos oficiales del mercado de vivienda para aportar contexto sobre la actividad inmobiliaria. Las tablas se descargan del [Boletín Online del MIVAU](https://apps.fomento.gob.es/BoletinOnline2/?nivel=2&orden=34000000), apartado *Transacciones inmobiliarias (compraventa)*:

| Tabla | Contenido | Uso en el TFM |
|---|---|---|
| 2.3 | Transacciones de vivienda nueva por municipios | Actividad del mercado local |
| 2.4 | Transacciones de vivienda de segunda mano por municipios | Actividad del mercado local |
| 3.3.1 | Valor medio de las transacciones de vivienda libre nueva (provincias) | Referencia de precio de transacción |
| 3.3.2 | Valor medio de las transacciones de vivienda libre de segunda mano (provincias) | Referencia de precio de transacción |

Esta fuente permite complementar los precios de oferta observados en los anuncios con indicadores del mercado real de compraventas: número de compraventas del municipio en los cuatro últimos trimestres publicados, su ratio por cada 1.000 habitantes y el valor medio de compraventa de la provincia.

### Instituto Nacional de Estadística (INE)

Se utilizan las *Cifras oficiales de población de los municipios españoles: Revisión del Padrón Municipal* ([detalle municipal](https://www.ine.es/dynt3/inebase/es/index.htm?padre=525)), una tabla por provincia, con la población a 1 de enero de 2025. Además de la población, aportan el **código INE** de cada municipio, que se utiliza como clave para cruzar Fotocasa, INE y MIVAU, ya que los nombres de municipio no coinciden siempre entre fuentes.

Del *Atlas de Distribución de Renta de los Hogares* (ADRH) se toma la **renta neta media por persona** de cada municipio en 2024, el último año publicado. INEbase ofrece el detalle municipal como un único fichero nacional (tabla 30824, *Indicadores de renta media y mediana* por municipios, distritos y secciones; unos 400 MB). Se reduce a las filas municipales de Andalucía con:

```bash
python src/variables_extra.py --extraer-renta ruta/a/30824.csv
```

que genera `data/externos/ine/renta/adrh_andalucia_municipios.csv` (3 MB). El fichero original no hace falta conservarlo.

### Línea de costa

`data/externos/geo/costa_andalucia.geojson` contiene la línea de costa del litoral andaluz y los tramos limítrofes del Algarve y Murcia, sin la costa africana. Procede de Natural Earth (escala 1:10 millones, paquete npm `world-atlas` 2.0.2), recortada y simplificada. Se usa para calcular la distancia de cada vivienda a la costa. Al ser un fichero pequeño de dominio público, se incluye en el repositorio.

### Instituto de Estadística y Cartografía de Andalucía (IECA)

El IECA constituye una fuente adicional para variables municipales y territoriales de Andalucía, incluyendo información demográfica, socioeconómica y geográfica.

### Otras fuentes

En función de la disponibilidad y utilidad metodológica, podrán incorporarse otras fuentes públicas relacionadas con turismo, territorio, catastro o accesibilidad geográfica.

## Adquisición de datos de Fotocasa

La adquisición de anuncios se realiza mediante Python utilizando principalmente Selenium y BeautifulSoup.

El proceso está dividido en dos fases.

### 1. Extracción de listados

Los scripts provinciales navegan por las páginas de resultados y realizan, entre otras, las siguientes tareas:

- descubrimiento automático de municipios;
- subdivisión de búsquedas grandes en zonas o subzonas;
- navegación de contenido dinámico con Selenium;
- scroll progresivo para capturar anuncios cargados dinámicamente;
- acumulación de anuncios aunque el DOM virtualice tarjetas;
- deduplicación por identificador de anuncio o URL;
- almacenamiento periódico de checkpoints;
- reanudación de ejecuciones interrumpidas;
- reinicio periódico del navegador para reducir problemas de estabilidad;
- registro de errores técnicos;
- diagnóstico de cobertura comparando, cuando es posible, los resultados declarados por el portal con los anuncios observados.

### 2. Enriquecimiento de detalle

Posteriormente se visita la ficha individual de cada anuncio para obtener variables adicionales que no siempre aparecen en los listados.

Antes de aceptar una ficha como válida se comprueba que corresponde al anuncio esperado mediante distintas señales, entre ellas:

- identificador del anuncio;
- título;
- precio;
- superficie;
- número de habitaciones.

También se detectan expresamente los anuncios que han dejado de estar disponibles para evitar mezclar datos del anuncio original con inmuebles recomendados por el portal.

### 3. Extracción específica de capitales de provincia

En la extracción provincial, cuando transcurre tiempo entre el listado y la consulta de detalle, una parte de los anuncios ya no está disponible al visitar su ficha. Además, las capitales, al dividirse en muchas zonas, eran las más expuestas a fallos del navegador, por lo que quedaron poco representadas.

Para ellas se utiliza un scraper específico (`scraper_capitales.py`) que:

- recorre las zonas (distritos) de cada capital;
- extrae el listado de una zona y visita **inmediatamente** las fichas de detalle de sus anuncios, de modo que apenas transcurren minutos entre ambas fases;
- reparte la muestra entre zonas mediante un cupo por zona, para no concentrarla en los primeros distritos recorridos;
- fija un objetivo de anuncios con detalle validado por capital;
- reinicia el navegador y reintenta la misma zona si la sesión se cae;
- guarda cada anuncio en el momento y permite reanudar la ejecución.

La salida tiene exactamente las mismas columnas que la del scraper de detalle, por lo que ambas se combinan después en un único conjunto de datos.

## Integración de fuentes

Los scripts de `src` preparan el conjunto de datos de análisis:

1. `procesar_mivau.py`: convierte las tablas Excel del MIVAU en CSV ordenados. Cada fichero se identifica por el número de tabla que aparece en su título.
2. `combinar_fotocasa.py`: une los anuncios con detalle de la extracción provincial y de la de capitales, eliminando duplicados (prevalece el detalle validado y, a igualdad, la extracción más reciente).
3. `enriquecer_fotocasa.py`: asigna a cada anuncio el código INE de su municipio y añade población, actividad del mercado y valor medio de transacción. Genera además una tabla de correspondencia entre municipios de Fotocasa e INE para documentar el cruce.

Algunas decisiones de integración:

- Las zonas o urbanizaciones que Fotocasa presenta como municipio (por ejemplo, Sotogrande o Montequinto) se asignan a su municipio oficial.
- Como referencia del MIVAU se usa el valor medio de vivienda nueva cuando el anuncio es de obra nueva o tiene menos de un año, y el de segunda mano en el resto.
- No se incorporan datos del Catastro: la referencia catastral aparece en una proporción mínima de anuncios, y cruzar por proximidad podría asignar a una vivienda los datos de otra cercana.

## Limpieza

`src/limpieza.py` genera el conjunto de datos de análisis a partir del enriquecido. Aplica, en orden, las siguientes reglas y registra en `informe_limpieza.csv` cuántos anuncios elimina cada una:

| # | Regla | Motivo |
|---|---|---|
| 1 | Solo anuncios con detalle validado | Criterio acordado con el tutor |
| 2 | Municipio identificado (código INE) | Elimina un pseudo-municipio de navegación del portal |
| 3 | Sin fincas rústicas | Superficies y precios no comparables con vivienda urbana |
| 4 | Sin subastas, nudas propiedades ni viviendas ocupadas | El precio no refleja el mercado libre |
| 5 | Sin viviendas ocupadas o sin posesión según el texto del anuncio | Mismo criterio que la regla 4, cuando la situación solo consta en la descripción |
| 6 | Con precio y superficie | Variable objetivo y principal predictora |
| 7 | Precio entre 20.000 € y 5.000.000 € | Errores de precio y casos singulares |
| 8 | Superficie entre 15 y 1.500 m² | Superficies imposibles o de parcela |
| 9 | Precio por m² entre 200 y 20.000 € | Incoherencias entre precio y superficie |
| 10 | Hasta 15 habitaciones y 10 baños | Valores imposibles |
| 11 | Sin duplicados | Misma vivienda publicada varias veces (mismas características y descripción) |

El resultado son 6.763 viviendas.

Además, normaliza las variables:

- **Equipamiento**: una característica se considera presente si figura en el listado o en la ficha de detalle, ya que cada fuente la recoge de forma incompleta.
- **Planta**: se convierte a número (bajo = 0, sótano = −1) aceptando solo formas inequívocas; el valor "+15 planta", que el portal usa como relleno, se trata como desconocido.
- **Estado, antigüedad y orientación**: los valores no informados se codifican como categoría propia ("Desconocido"); la antigüedad se resume además en años (punto medio del intervalo).
- **Tipo de vivienda**: los anuncios sin tipo se clasifican a partir del título.

Las columnas `precio_m2` y `bajada_precio_eur` se conservan como informativas, pero derivan del precio y no deben usarse como predictoras.

## Análisis exploratorio

`notebooks/01_analisis_exploratorio.ipynb` analiza el conjunto de datos limpio y guarda sus figuras en `doc/figuras/eda/` para la memoria. Principales conclusiones:

- El precio tiene una fuerte asimetría (coeficiente 4,4) que el logaritmo corrige casi por completo (0,1), por lo que se modelará `log_precio`.
- La provincia no basta para explicar el precio: en todas salvo Málaga la capital es más cara que el resto, y la costa concentra los precios más altos.
- Pisos y casas difieren en nivel de precio por m², pero la relación entre precio y superficie tiene casi la misma pendiente en ambos grupos, lo que permite un único modelo con el tipo de vivienda como variable.
- Las variables más asociadas al precio son los baños, el valor medio de compraventa del MIVAU y la superficie; la actividad del mercado y la población aportan información de localización.
- La antigüedad tiene una relación no lineal con el precio (los pisos de cascos históricos vuelven a encarecerse), lo que favorece a los modelos basados en árboles.

## Modelado

`src/preprocesado.py` fija las variables del modelo y una partición entrenamiento/prueba 80/20 estratificada por provincia (`data/procesado/particion.csv`): 5.401 viviendas de entrenamiento y 1.362 de prueba. La variable objetivo es `log_precio`. Las variables que describen el anuncio y no la vivienda (número de fotos, longitud de la descripción, bajada de precio) se excluyen del modelo principal y se evalúan aparte.

`src/modelado.py` ajusta los hiperparámetros de cada modelo con Optuna y evalúa el modelo final una única vez sobre prueba:

- **Búsqueda:** muestreador TPE, 40 pruebas por modelo, validación cruzada de 5 particiones sobre entrenamiento.
- **Criterio de ajuste y selección:** RMSE de log(precio), que es la función de pérdida con la que se entrenan los modelos.
- **Métricas reportadas:** todas en la escala original (euros), tras revertir la transformación logarítmica: MAE, RMSE, MAPE, MdAPE y R². Se calculan tanto en validación cruzada como en prueba.
- **Rangos de búsqueda:** cada ejecución los guarda en `resultados/espacios_busqueda.csv`, generados del propio código. El notebook 02 los muestra junto al valor elegido para cada modelo.

| Modelo | MAE (CV) | R² (CV) | MAE (prueba) | RMSE (prueba) | MdAPE (prueba) | R² (prueba) |
|---|---|---|---|---|---|---|
| **LightGBM** | **78.091 €** | **0,752** | **81.199 €** | **180.857 €** | **16,4 %** | **0,738** |
| XGBoost | 78.642 € | 0,743 | 79.169 € | 177.459 € | 16,3 % | 0,748 |
| CatBoost | 78.863 € | 0,752 | 82.957 € | 182.761 € | 16,9 % | 0,732 |
| Random Forest | 83.405 € | 0,714 | 85.821 € | 195.224 € | 17,6 % | 0,695 |
| Regresión lineal | 101.042 € | 0,644 | 104.681 € | 220.438 € | 23,0 % | 0,611 |

Los tres modelos de *boosting* están prácticamente empatados. LightGBM se selecciona por tener el menor error en validación cruzada. El R² en euros es moderado porque lo dominan las pocas viviendas de más de un millón de euros; por eso se acompaña de métricas relativas (MdAPE).

`notebooks/02_modelado.ipynb` analiza los resultados:

- **Hiperparámetros:** tabla de rangos y valores elegidos, y convergencia de Optuna. Los valores que quedan en el borde de su rango son límites naturales, como regularización nula o `min_samples_leaf = 1`.
- **Segmentos:** el error es mayor en casas (19,9 %) que en pisos (14,5 %), y apenas varía entre capitales y resto de municipios. La principal limitación es la regresión hacia la media: se sobrestiman las viviendas más baratas (+16 %) y se infravaloran las más caras (−15 %).
- **Variables del anuncio:** no mejoran el modelo, lo que confirma su exclusión.
- **Explicabilidad (SHAP):** la superficie, los baños, la ubicación, el valor medio del MIVAU, la piscina y el garaje son las variables más influyentes. El tipo de vivienda apenas contribuye por sí mismo: la diferencia entre casas y pisos se recoge a través de la parcela, el jardín o la localización.

Para reproducir:

```bash
python src/preprocesado.py
python src/modelado.py --pruebas 40
```

### Variables adicionales para los extremos de precio

Para corregir la regresión hacia la media, `src/variables_extra.py` define tres grupos de variables que se activan con `preparar(grupos=...)`:

- **texto:** 13 indicadores extraídos del título y la descripción mediante expresiones regulares: vistas al mar, primera línea, playa, lujo, piscina desbordante, golf, spa, ático, a estrenar, reformado, a reformar, cueva e inversión. Se eligieron analizando los residuos fuera de muestra del modelo base, solo en entrenamiento.
- **geo:** distancia a la costa y a la capital de provincia, calculadas a partir de las coordenadas.
- **renta:** renta neta media por persona del municipio (INE, ADRH).

```bash
python src/modelado.py --pruebas 40 --grupos texto geo renta   # salida en resultados/texto_geo_renta/
python src/ablacion_variables.py --grupos texto geo renta      # aportación de cada grupo
```

### Redes neuronales

`src/redes_neuronales.py` evalúa dos usos de redes neuronales sobre el mejor modelo ampliado:

- **MLP** con las mismas variables. No supera a LightGBM (MAE en validación cruzada de 83.621 € frente a 72.296 €), y combinarlo con LightGBM tampoco aporta, porque sus errores están correlados en un 0,85.
- **Corrección por texto.** Un modelo de lenguaje multilingüe (`multilingual-e5-small`) convierte el título y la descripción en un vector de 384 dimensiones (`src/embeddings_texto.py`). A partir de ese vector, una regresión ridge predice el error del modelo tabular y lo corrige. Mejora en las 5 particiones de validación cruzada y, sobre todo, en las viviendas caras.

```bash
python -m pip install sentence-transformers   # solo para calcular los embeddings
python src/embeddings_texto.py                # una vez, ~10 min en CPU
python src/redes_neuronales.py                # ~3 min
```

### Resultados de la mejora

`notebooks/03_mejora_variables.ipynb` reúne la comparación, la ablación y las redes neuronales:

| LightGBM | MAE (CV) | R² (CV) | MAE (prueba) | RMSE (prueba) | MdAPE (prueba) | R² (prueba) |
|---|---|---|---|---|---|---|
| Modelo base | 78.091 € | 0,752 | 81.199 € | 180.857 € | 16,4 % | 0,738 |
| Modelo ampliado | 72.296 € | 0,783 | 74.836 € | 173.765 € | 14,7 % | 0,758 |
| **Ampliado + corrección por texto (final)** | **71.551 €** | **0,797** | **73.322 €** | **165.888 €** | **15,1 %** | **0,780** |

- **Por grupo de variables:** geografía y texto reducen el MAE de validación cruzada en torno a un 4 % y un 3 % respectivamente, y sus efectos se suman. La renta municipal aporta poco (≈1 %), porque es en gran parte redundante con el contexto municipal que ya tenía el modelo.
- **Distancia a la costa:** es la cuarta variable más importante. Estar a menos de 1 km supone en torno a un +17 % de precio.
- **Corrección por texto:** mejora sobre todo las viviendas caras, cuyo sesgo pasa de −13,6 % a −11,1 %. En los tramos centrales la MdAPE empeora ligeramente.
- **Limitación que persiste:** la regresión hacia la media se reduce, pero sigue siendo notable (en torno a ±12 % en los extremos). Depende de información que el anuncio no describe, como calidades, reformas o el margen de negociación.

## Código de referencia

La primera implementación del proceso de adquisición de datos se basó en el repositorio público de Rocío Benítez:

[rociobenitez/fotocasa-scraper](https://github.com/rociobenitez/fotocasa-scraper)

Dicho proyecto proporcionó una base inicial para trabajar con Selenium y BeautifulSoup sobre contenido de Fotocasa, incluyendo navegación automatizada, scroll y extracción de atributos básicos de los anuncios.

A partir de esa referencia, el código de este repositorio se ha ampliado y reorganizado para adaptarlo a las necesidades específicas del TFM. Entre las principales ampliaciones se encuentran:

- cobertura de las ocho provincias andaluzas;
- descubrimiento y procesamiento de municipios y zonas;
- subdivisión recursiva de búsquedas grandes;
- gestión de checkpoints y reanudación;
- deduplicación acumulativa;
- reinicio y recuperación ante fallos del navegador;
- separación entre scraping de listados y scraping de detalle;
- validación de identidad de las fichas individuales;
- detección de anuncios retirados;
- extracción de variables adicionales;
- generación de métricas de cobertura y completitud;
- preparación de un conjunto de datos final orientado al análisis estadístico y al aprendizaje automático.

Por tanto, este repositorio no constituye una copia directa del proyecto de referencia, sino una adaptación y extensión desarrollada específicamente para los objetivos metodológicos de este TFM.

## Estructura del repositorio

```text
tfm-vivienda-andalucia/
├── scrapers/                         Adquisición de datos de Fotocasa
│   ├── fotocasa_base.py              lógica común del scraper de listados
│   ├── provincias.py                 provincias y URL de partida
│   ├── scraper_fotocasa_<provincia>.py   un lanzador por provincia
│   ├── scraper_fotocasa_andalucia.py     lanza las ocho provincias
│   ├── scraper_detalle_fotocasa.py   fichas de detalle y validación
│   └── scraper_capitales.py          capitales: listado + detalle a la vez
│
├── src/                              Preparación del conjunto de datos
│   ├── procesar_mivau.py
│   ├── combinar_fotocasa.py
│   ├── enriquecer_fotocasa.py
│   ├── limpieza.py
│   ├── preprocesado.py               variables y partición entrenamiento/prueba
│   ├── variables_extra.py            variables de texto, geografía y renta
│   ├── modelado.py                   ajuste (Optuna) y evaluación de modelos
│   ├── ablacion_variables.py         aportación de cada grupo de variables
│   ├── embeddings_texto.py           representación del texto con un modelo de lenguaje
│   └── redes_neuronales.py           MLP y corrección por texto
│
├── data/                             (contenido excluido del repositorio)
│   ├── provincias/                   listados provinciales
│   ├── capitales/                    extracción de capitales
│   ├── externos/
│   │   ├── mivau/                    Excel descargados del MIVAU
│   │   ├── ine/                      Excel descargados del INE (renta/: ADRH)
│   │   └── geo/                      línea de costa (sí se versiona)
│   └── procesado/                    salidas de src/
│
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb
│   ├── 02_modelado.ipynb
│   └── 03_mejora_variables.ipynb
│
├── doc/                              memoria del TFM (LaTeX)
│   └── figuras/
│       ├── eda/                      figuras del análisis exploratorio
│       ├── modelado/                 figuras de resultados y explicabilidad
│       └── mejora/                   figuras del modelo ampliado
├── models/                           modelos entrenados (excluidos del repositorio)
├── .gitignore
├── requirements.txt
└── README.md
```

La carpeta `scrapers` contiene el código de adquisición de datos y `src` el de integración. Las fases posteriores del TFM (limpieza, análisis exploratorio, generación de variables y modelado) se incorporarán progresivamente en `src` y `notebooks`.

## Datos no incluidos en el repositorio

Los ficheros CSV obtenidos mediante scraping de Fotocasa **no se publican en este repositorio**.

El objetivo del repositorio es hacer reproducible el código y documentar la metodología, no redistribuir el contenido obtenido del portal inmobiliario. Por esta razón, los datos extraídos, checkpoints, ficheros de estado y otros resultados de las ejecuciones se encuentran excluidos mediante `.gitignore`.

Esta decisión permite separar claramente:

- el código desarrollado para el TFM;
- las fuentes públicas que pueden descargarse de sus organismos oficiales;
- los datos obtenidos mediante scraping, que se mantienen fuera del repositorio público.

## Requisitos

Se recomienda utilizar Python 3.10 o superior.

Las principales dependencias son `selenium`, `beautifulsoup4` y `pandas` para la adquisición, y `xlrd` y `openpyxl` para leer los Excel de las fuentes oficiales.

La instalación completa puede realizarse mediante:

```bash
pip install -r requirements.txt
```

También es necesario disponer de Google Chrome. Las versiones recientes de Selenium pueden gestionar automáticamente el controlador compatible mediante Selenium Manager.

## Ejecución

### Ejecutar una provincia

Desde la carpeta `scrapers`:

```bash
python scraper_fotocasa_malaga.py
```

El mismo esquema puede utilizarse para las demás provincias.

### Ejecutar todas las provincias

```bash
python scraper_fotocasa_andalucia.py
```

### Ejecutar el enriquecimiento de detalle

Una vez generados los ficheros provinciales:

```bash
python scraper_detalle_fotocasa.py
```

El scraper de detalle combina los anuncios provinciales, elimina duplicados y visita las fichas individuales para completar el conjunto de datos.

### Ejecutar la extracción de capitales

```bash
python scraper_capitales.py                   # las ocho capitales
python scraper_capitales.py malaga sevilla    # solo las indicadas
```

Las capitales que ya alcanzan el objetivo de anuncios validados (`OBJETIVO_VALIDOS`) se omiten. La ejecución puede interrumpirse y relanzarse: continúa por la zona en la que se quedó.

### Preparar el conjunto de datos

Tras colocar los Excel del MIVAU en `data/externos/mivau/` y los del INE en `data/externos/ine/`, desde la raíz del repositorio:

```bash
python src/procesar_mivau.py
python src/combinar_fotocasa.py
python src/enriquecer_fotocasa.py
python src/limpieza.py
```

El resultado es `data/procesado/fotocasa_andalucia_limpio.csv`.

## Metodología de análisis prevista

Tras la adquisición e integración de datos, el flujo metodológico del TFM contempla las siguientes etapas:

1. Control de calidad de los datos.
2. Tratamiento de valores ausentes y posibles anomalías.
3. Eliminación de duplicados.
4. Homogeneización de variables y unidades.
5. Integración con fuentes externas oficiales.
6. Análisis exploratorio de datos.
7. Ingeniería y selección de variables.
8. Definición de conjuntos de entrenamiento, validación y prueba.
9. Entrenamiento de modelos predictivos.
10. Optimización de hiperparámetros.
11. Evaluación comparativa de modelos.
12. Interpretación de resultados.

Entre los modelos inicialmente considerados se encuentran:

- regresión lineal como baseline;
- Random Forest;
- XGBoost;
- LightGBM;
- CatBoost;
- de forma opcional, una red neuronal multicapa (MLP).

Para la optimización de hiperparámetros se contempla el uso de Optuna y para la interpretación de modelos se estudiará el uso de SHAP.

## Métricas de evaluación

Todas las métricas se calculan en la escala original del precio (euros), tras revertir la transformación logarítmica con la que se entrenan los modelos:

- Mean Absolute Error (MAE);
- Root Mean Squared Error (RMSE);
- coeficiente de determinación (R²);
- error porcentual absoluto medio y mediano (MAPE y MdAPE), que pesan igual a todas las viviendas con independencia de su precio.

La predicción en euros se obtiene como la exponencial de la predicción en logaritmo, que estima la mediana del precio condicionada a las variables.

La selección del modelo final no dependerá únicamente del error predictivo. También se tendrá en cuenta la estabilidad del modelo, su capacidad de generalización y la interpretabilidad de los factores asociados al precio.

## Reproducibilidad

Uno de los objetivos del proyecto es mantener la trazabilidad completa del proceso. Para ello se documentarán:

- origen de cada fuente;
- fecha de extracción o consulta;
- variables utilizadas;
- reglas de limpieza;
- transformaciones realizadas;
- criterios de exclusión;
- estrategia de partición de los datos;
- configuración de modelos e hiperparámetros;
- métricas obtenidas.

El código fuente se mantendrá versionado mediante Git.

## Consideraciones y limitaciones

El proyecto presenta varias limitaciones que deben tenerse en cuenta al interpretar los resultados:

- Los precios publicados en Fotocasa son precios de oferta y no precios finales de compraventa.
- La oferta observada en un portal inmobiliario no representa necesariamente la totalidad del mercado residencial.
- Algunos anuncios pueden desaparecer entre la extracción del listado y la consulta de detalle.
- Determinadas características de los inmuebles pueden estar ausentes o ser declaradas de forma no homogénea por los anunciantes.
- Las coordenadas proporcionadas por los portales inmobiliarios pueden ser aproximadas y no deben interpretarse necesariamente como la dirección exacta del inmueble.
- Las fuentes externas pueden presentar diferentes frecuencias temporales y niveles de agregación territorial.
- El número de anuncios observados puede verse condicionado por la forma en que el portal carga y presenta sus resultados.

Estas limitaciones serán documentadas y consideradas tanto en la preparación del conjunto de datos como en la discusión final del trabajo.

## Uso responsable

El código de scraping se ha desarrollado exclusivamente con fines académicos y de investigación dentro del contexto del TFM.

No se incluyen mecanismos destinados a superar CAPTCHAs, controles de acceso o medidas explícitas de bloqueo. Si el portal limita o impide una determinada consulta, el proceso debe registrar la incidencia y detener o continuar la ejecución de forma controlada.

La publicación de este repositorio tiene como finalidad documentar la metodología y compartir el código desarrollado, no redistribuir el contenido obtenido del portal.

## Autor

Mario Aroca Páez

Trabajo Fin de Máster  
Máster Universitario en Ciencia de Datos  
Universitat Oberta de Catalunya (UOC)

## Referencias técnicas

- [Repositorio de referencia: rociobenitez/fotocasa-scraper](https://github.com/rociobenitez/fotocasa-scraper)
- [Selenium](https://www.selenium.dev/documentation/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [pandas](https://pandas.pydata.org/docs/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Optuna](https://optuna.org/)
- [SHAP](https://shap.readthedocs.io/)

## Estado del proyecto

Proyecto en desarrollo.

- [x] Extracción de listados y fichas de detalle de las ocho provincias.
- [x] Integración con MIVAU e INE mediante código de municipio.
- [x] Ampliación de la muestra de las capitales de provincia.
- [x] Limpieza del conjunto de datos.
- [x] Análisis exploratorio.
- [x] Entrenamiento, evaluación e interpretación de modelos.
- [x] Variables adicionales (texto, geografía, renta) y ablación.
- [x] Redes neuronales: MLP y corrección por texto con un modelo de lenguaje.
