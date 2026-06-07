# clustering_transporte.py
# Script sencillo para generar un dataset sintético y aplicar K-Means (k=3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend no interactivo para generar imágenes en entorno servidor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generate_dataset(csv_path='dataset_transporte.csv', n=100, random_state=42):
    """Genera un dataset sintético realista y lo guarda como CSV.

    Cada estación tendrá: pasajeros (por día), tiempo_espera (min), transbordos (conteo).
    Se simulan tres perfiles (baja/media/alta afluencia) para que K-Means pueda distinguirlos.
    """
    np.random.seed(random_state)
    stations = [f'Estacion_{i:03d}' for i in range(1, n+1)]

    # Distribuciones intencionales para crear grupos detectables
    sizes = [20, 50, 30]  # baja, media, alta (suma = n)
    assert sum(sizes) == n

    pasajeros = []
    tiempo_espera = []
    transbordos = []

    for idx, size in enumerate(sizes):
        if idx == 0:  # baja afluencia
            p = np.random.normal(500, 150, size)
            t = np.random.normal(12, 3, size)
            tr = np.random.poisson(0.6, size)
        elif idx == 1:  # media afluencia
            p = np.random.normal(3000, 800, size)
            t = np.random.normal(6, 2, size)
            tr = np.random.poisson(1.0, size)
        else:  # alta afluencia
            p = np.random.normal(12000, 2500, size)
            t = np.random.normal(3, 1, size)
            tr = np.random.poisson(1.5, size)

        pasajeros.extend(p.tolist())
        tiempo_espera.extend(t.tolist())
        transbordos.extend(tr.tolist())

    # Asegurar valores válidos y formato
    pasajeros = np.clip(pasajeros, 50, None).astype(int)
    tiempo_espera = np.clip(tiempo_espera, 0.5, None)
    transbordos = np.clip(transbordos, 0, None).astype(int)

    df = pd.DataFrame({
        'estacion': stations,
        'pasajeros': pasajeros,
        'tiempo_espera': np.round(tiempo_espera, 2),
        'transbordos': transbordos
    })

    df.to_csv(csv_path, index=False)
    return df


def eda(df, results_dir='resultados'):
    """Exploración básica: estadísticas, faltantes y gráficos simples.
    Los gráficos se guardan en `results_dir`.
    """
    os.makedirs(results_dir, exist_ok=True)

    print('Primeras filas:')
    print(df.head().to_string(index=False))

    print('\nEstadísticas descriptivas:')
    print(df.describe().to_string())

    print('\nValores faltantes por columna:')
    print(df.isnull().sum().to_string())

    print('\nTipos de datos:')
    print(df.dtypes)

    # Histograma de pasajeros
    plt.figure(figsize=(8, 4))
    plt.hist(df['pasajeros'], bins=20, color='C0', edgecolor='k')
    plt.title('Histograma de pasajeros por estación')
    plt.xlabel('Pasajeros (por día)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hist_pasajeros.png'))
    plt.close()

    # Histograma de tiempo de espera
    plt.figure(figsize=(8, 4))
    plt.hist(df['tiempo_espera'], bins=20, color='C1', edgecolor='k')
    plt.title('Histograma de tiempo de espera (minutos)')
    plt.xlabel('Tiempo de espera (min)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hist_tiempo_espera.png'))
    plt.close()

    # Diagrama de dispersión pasajeros vs tiempo_espera
    plt.figure(figsize=(7, 5))
    plt.scatter(df['pasajeros'], df['tiempo_espera'], alpha=0.7)
    plt.xscale('log')  # escala log para mejor visualización de rango amplio
    plt.xlabel('Pasajeros (escala log)')
    plt.ylabel('Tiempo de espera (min)')
    plt.title('Pasajeros vs Tiempo de espera')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'scatter_pasajeros_espera.png'))
    plt.close()


def prepare_and_cluster(df, results_dir='resultados', random_state=42):
    """Prepara variables numéricas, aplica escalamiento y K-Means (k=3).
    Agrega columna `cluster` al DataFrame y guarda gráfico de clusters.
    """
    features = ['pasajeros', 'tiempo_espera', 'transbordos']
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print('\nDatos escalados (primeras filas):')
    print(pd.DataFrame(X_scaled, columns=features).head().to_string(index=False))

    # K-Means con k=3
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    df['cluster'] = labels

    # Invertir centroides a escala original para mostrar en la gráfica
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    # Gráfico de dispersión coloreado por cluster
    plt.figure(figsize=(8, 6))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for cluster in range(k):
        mask = df['cluster'] == cluster
        plt.scatter(df.loc[mask, 'pasajeros'], df.loc[mask, 'tiempo_espera'],
                    c=colors[cluster], label=f'Cluster {cluster}', alpha=0.7, edgecolors='k', linewidths=0.3)

    # Centroides
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centroides')
    plt.xscale('log')
    plt.xlabel('Pasajeros (escala log)')
    plt.ylabel('Tiempo de espera (min)')
    plt.title('Clusters de estaciones (K-Means k=3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'clusters.png'))
    plt.close()

    # Imprimir conteos y centroides
    counts = df['cluster'].value_counts().sort_index()
    print('\nCantidad de estaciones por cluster:')
    print(counts.to_string())

    print('\nCentroides (escala original):')
    centers_df = pd.DataFrame(centers, columns=features)
    centers_df['cluster'] = range(k)
    print(centers_df.to_string(index=False))

    print('\nMedias por cluster:')
    print(df.groupby('cluster')[features].mean().to_string())

    return df


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_dir, 'dataset_transporte.csv')
    results_dir = os.path.join(project_dir, 'resultados')

    # Generar dataset si no existe
    if not os.path.exists(csv_path):
        print('Generando dataset sintético...')
        df = generate_dataset(csv_path=csv_path, n=100, random_state=42)
    else:
        df = pd.read_csv(csv_path)

    # EDA básico
    eda(df, results_dir=results_dir)

    # Preparación y clustering
    df = prepare_and_cluster(df, results_dir=results_dir, random_state=42)

    # Guardar dataset final con columna cluster
    df.to_csv(csv_path, index=False)
    print('\nClustering final guardado en:', csv_path)

    # Interpretación breve
    print('\nInterpretación breve de clusters:')
    means = df.groupby('cluster')[['pasajeros', 'tiempo_espera', 'transbordos']].mean()
    for idx, row in means.iterrows():
        print(f'Cluster {idx}: pasajeros={row["pasajeros"]:.1f}, tiempo_espera={row["tiempo_espera"]:.2f} min, transbordos={row["transbordos"]:.2f}')
        if row['pasajeros'] < 2000:
            print('  - Estaciones de baja afluencia y tiempos de espera mayores.')
        elif row['pasajeros'] < 8000:
            print('  - Estaciones de afluencia media y tiempos de espera moderados.')
        else:
            print('  - Estaciones de alta afluencia y tiempos de espera cortos.')

    # Conclusiones (máx. 5 párrafos, impresos)
    conclusions = [
        'K-Means permite segmentar estaciones con patrones similares según las variables seleccionadas.',
        'La normalización con StandardScaler es necesaria para que variables con escalas distintas no dominen el clustering.',
        'Los clusters resultantes pueden orientar decisiones operativas, p.ej., priorizar frecuencias en estaciones con alta demanda.',
        'Este enfoque es simple y rápido; útil como análisis exploratorio antes de aplicar métodos más sofisticados.',
        'Mejoras futuras: añadir series temporales, variables adicionales y comparar con otros algoritmos de clustering.'
    ]
    print('\nConclusiones:')
    for c in conclusions:
        print('-', c)


if __name__ == '__main__':
    main()
