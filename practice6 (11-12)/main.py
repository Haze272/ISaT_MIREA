from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import umap
import trimap
import pacmap
import os

os.environ['OMP_NUM_THREADS'] = '1'

def calculate_ari_scores(model_class, X, y, min_clusters=2, max_clusters=11):
    ari_scores = {}
    for n_clusters in range(min_clusters, max_clusters):
        model = model_class(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(X)
        ari = adjusted_rand_score(y, cluster_labels)
        ari_scores[n_clusters] = ari
    return ari_scores


def visualize_clusters(X, y, cluster_model, dim_reduction_model, ax, **kwargs):
    X_reduced = dim_reduction_model.fit_transform(X)
    labels = cluster_model.fit_predict(X)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        ax.scatter(X_reduced[labels == label, 0], X_reduced[labels == label, 1],
                   color=colors(i), label=f'Cluster {label}', alpha=0.5)

    if isinstance(cluster_model, KMeans):
        centroids = cluster_model.cluster_centers_
        centroids_reduced = dim_reduction_model.transform(centroids) if hasattr(dim_reduction_model, 'transform') else None

        if centroids_reduced is not None:
            ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1],
                       color='black', marker='X', s=200, label='Centroids')

    ax.set_title(f'Clusters Visualization with {dim_reduction_model.__class__.__name__}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()


def evaluate_agglomerative_clustering(X, n_clusters_range=[2, 11]):
    silhouettes = []

    for n_clusters in n_clusters_range:
        agglom = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agglom.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouettes.append(silhouette_avg)

    return silhouettes


def evaluate_k_means(X, n_clusters_range=[2, 11]):
    sse = []
    silhouettes = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        cluster_labels = kmeans.predict(X)

        sse.append(kmeans.inertia_)

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouettes.append(silhouette_avg)

    return sse, silhouettes


def plot_all_results(n_clusters_range, agg_silhouettes, km_sse, km_silhouettes):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    axs[0][0].plot(n_clusters_range, agg_silhouettes, marker='o')
    axs[0][0].set_xlabel('Количество кластеров (K)')
    axs[0][0].set_ylabel('Средний индекс силуэта')
    axs[0][0].set_title('Изменение среднего индекса силуэта\nдля Agglomerative Clustering')
    axs[0][0].grid(True)

    axs[1][0].plot(n_clusters_range, km_sse, marker='o')
    axs[1][0].set_xlabel('Количество кластеров (K)')
    axs[1][0].set_ylabel('Сумма квадратов ошибок (SSE)')
    axs[1][0].set_title('График "метода локтя" для KMeans')
    axs[1][0].grid(True)

    axs[1][1].plot(n_clusters_range, km_silhouettes, marker='o')
    axs[1][1].set_xlabel('Количество кластеров (K)')
    axs[1][1].set_ylabel('Средний индекс силуэта')
    axs[1][1].set_title('Изменение среднего индекса силуэта\nдля KMeans')
    axs[1][1].grid(True)

    axs[0][1].axis('off')

    plt.tight_layout()
    plt.show()



def main():
    # ЗАДАНИЕ 1
    glass = fetch_ucirepo(id=52)

    X = glass.data.features.dropna()
    y = glass.data.targets.values.flatten()
    y = y[X.index]

    agglo_ari_scores = calculate_ari_scores(AgglomerativeClustering, X, y)

    kmeans_ari_scores = calculate_ari_scores(KMeans, X, y)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(list(agglo_ari_scores.keys()), list(agglo_ari_scores.values()), marker='o', label="Agglomerative")
    ax[0].set_title('ARI for Agglomerative Clustering')
    ax[0].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('ARI Score')
    ax[0].legend()

    ax[1].plot(list(kmeans_ari_scores.keys()), list(kmeans_ari_scores.values()), marker='o', label="K-Means")
    ax[1].set_title('ARI for K-Means Clustering')
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('ARI Score')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    optimal_n_clusters = input("Оптимальное количество кластеров ")
    model_num = input("Выберите модель. 0 - AgglomerativeClustering, 1 - KMeans ")

    if model_num == 0:
        model = AgglomerativeClustering(n_clusters=int(optimal_n_clusters), random_state=42)
    else:
        model = KMeans(n_clusters=int(optimal_n_clusters), random_state=42)

    X_np = X.values
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    visualize_clusters(X_np, y, model, TSNE(n_components=2), axs[0, 0])
    visualize_clusters(X_np, y, model, umap.UMAP(n_components=2), axs[0, 1])
    visualize_clusters(X_np, y, model, trimap.TRIMAP(n_dims=2), axs[1, 0])
    visualize_clusters(X_np, y, model, pacmap.PaCMAP(save_tree=True), axs[1, 1])

    plt.tight_layout()
    plt.show()

    # ЗАДАНИЕ 2
    df = pd.read_csv(
        "/home/esteban/Documents/GitHub/ISaT_MIREA/practice6 (11-12)/archive/country_wise_latest.csv",
        sep=',',
        header=0
    )
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[-1], axis=1)
    df = df.drop(df.columns[-1], axis=1)
    df = df.drop(df.columns[-1], axis=1)
    df = df.drop(df.columns[-1], axis=1)

    target_column = df.columns[-1]

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    n_clusters_range = list(range(2, 11))
    agg_silhouettes = evaluate_agglomerative_clustering(X, n_clusters_range)
    km_sse, km_silhouettes = evaluate_k_means(X, n_clusters_range)

    plot_all_results(n_clusters_range, agg_silhouettes, km_sse, km_silhouettes)

    optimal_n_clusters = input("Оптимальное количество кластеров ")
    model_num = input("Выберите модель. 0 - AgglomerativeClustering, 1 - KMeans ")

    if model_num == 0:
        model = AgglomerativeClustering(n_clusters=int(optimal_n_clusters), random_state=42)
    else:
        model = KMeans(n_clusters=int(optimal_n_clusters), random_state=42)

    X_np = X.values
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    visualize_clusters(X_np, y, model, TSNE(n_components=2), axs[0, 0])
    visualize_clusters(X_np, y, model, umap.UMAP(n_components=2), axs[0, 1])
    visualize_clusters(X_np, y, model, trimap.TRIMAP(n_dims=2), axs[1, 0])
    visualize_clusters(X_np, y, model, pacmap.PaCMAP(save_tree=True), axs[1, 1])

    plt.tight_layout()
    plt.show()

    # ЗАДАНИЕ 3
    df = pd.read_csv("/home/esteban/Documents/GitHub/ISaT_MIREA/practice2 (3-4)//mammoth.csv")

    df = df.sample(3000, random_state=42)
    target_column = df.columns[-1]

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    n_clusters_range = list(range(2, 11))
    agg_silhouettes = evaluate_agglomerative_clustering(X, n_clusters_range)
    km_sse, km_silhouettes = evaluate_k_means(X, n_clusters_range)

    plot_all_results(n_clusters_range, agg_silhouettes, km_sse, km_silhouettes)

    optimal_n_clusters = input("Оптимальное количество кластеров ")
    model_num = input("Выберите модель. 0 - AgglomerativeClustering, 1 - KMeans ")

    if model_num == 0:
        model = AgglomerativeClustering(n_clusters=int(optimal_n_clusters), random_state=42)
    else:
        model = KMeans(n_clusters=int(optimal_n_clusters), random_state=42)

    X_np = X.values
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    visualize_clusters(X_np, y, model, TSNE(n_components=2), axs[0, 0])
    visualize_clusters(X_np, y, model, umap.UMAP(n_components=2), axs[0, 1])
    visualize_clusters(X_np, y, model, trimap.TRIMAP(n_dims=2), axs[1, 0])
    visualize_clusters(X_np, y, model, pacmap.PaCMAP(save_tree=True), axs[1, 1])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
