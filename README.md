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

Se utilizan datos oficiales del mercado de vivienda para aportar contexto sobre la actividad inmobiliaria. Entre las variables disponibles se encuentran el número y el valor agregado de las transacciones inmobiliarias, desagregados por territorio, periodo y tipo de vivienda.

Esta fuente permite complementar los precios de oferta observados en los anuncios con indicadores del mercado real de compraventas.

### Instituto Nacional de Estadística (INE)

Se prevé incorporar información estadística relacionada con renta, población y otras características socioeconómicas que permitan caracterizar el entorno de los inmuebles.

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
├── scrapers/
│   ├── __init__.py
│   ├── fotocasa_base.py
│   ├── provincias.py
│   ├── scraper_detalle_fotocasa.py
│   ├── scraper_fotocasa_andalucia.py
│   ├── scraper_fotocasa_almeria.py
│   ├── scraper_fotocasa_cadiz.py
│   ├── scraper_fotocasa_cordoba.py
│   ├── scraper_fotocasa_granada.py
│   ├── scraper_fotocasa_huelva.py
│   ├── scraper_fotocasa_jaen.py
│   ├── scraper_fotocasa_malaga.py
│   └── scraper_fotocasa_sevilla.py
│
├── data/
│   └── provincias/
│       └── .gitkeep
│
├── notebooks/
│   └── .gitkeep
│
├── src/
│   └── .gitkeep
│
├── .gitignore
├── requirements.txt
└── README.md
```

La carpeta `scrapers` contiene el código de adquisición y enriquecimiento de datos. Las fases posteriores del TFM, como limpieza, integración, análisis exploratorio, generación de variables y modelado, se incorporarán progresivamente en `src` y `notebooks`.

## Datos no incluidos en el repositorio

Los ficheros CSV obtenidos mediante scraping de Fotocasa **no se publican en este repositorio**.

El objetivo del repositorio es hacer reproducible el código y documentar la metodología, no redistribuir el contenido obtenido del portal inmobiliario. Por esta razón, los datos extraídos, checkpoints, ficheros de estado y otros resultados de las ejecuciones se encuentran excluidos mediante `.gitignore`.

Esta decisión permite separar claramente:

- el código desarrollado para el TFM;
- las fuentes públicas que pueden descargarse de sus organismos oficiales;
- los datos obtenidos mediante scraping, que se mantienen fuera del repositorio público.

## Requisitos

Se recomienda utilizar Python 3.10 o superior.

Las principales dependencias son:

```text
pandas
selenium
beautifulsoup4
```

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
- árboles de decisión;
- Random Forest;
- XGBoost;
- LightGBM.

Para la optimización de hiperparámetros se contempla el uso de Optuna y para la interpretación de modelos se estudiará el uso de SHAP.

## Métricas de evaluación

Los modelos se compararán principalmente mediante:

- Mean Absolute Error (MAE);
- Root Mean Squared Error (RMSE);
- coeficiente de determinación (R²).

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

Actualmente se está trabajando en la adquisición, validación y enriquecimiento del conjunto de datos de vivienda en Andalucía. Las fases de integración con fuentes oficiales, análisis exploratorio y modelado se incorporarán progresivamente al repositorio.
