# Proyecto: Clustering en Sistema de Transporte Masivo

## Introducción
Proyecto educativo que muestra un ejemplo sencillo de aprendizaje no supervisado (K-Means) aplicado a un conjunto sintético de estaciones de transporte.

## Objetivo
Identificar grupos de estaciones con características similares usando K-Means (k=3).

## Tecnologías
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn

## Instalación
Crear un entorno e instalar dependencias:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Ejecución
Ejecutar el script principal:

```bash
python clustering_transporte.py
```

Se generarán los archivos:
- `dataset_transporte.csv` (dataset sintético)
- Imágenes en la carpeta `resultados/` (`hist_pasajeros.png`, `hist_tiempo_espera.png`, `scatter_pasajeros_espera.png`, `clusters.png`)

## Resultados
El script realiza EDA, aplica StandardScaler y K-Means (k=3). Los clusters se agregan como columna `cluster` en el CSV.

## Conclusiones
Ver las conclusiones impresas al final de la ejecución del script.
