from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import trimap
import pacmap
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_metrics, test_metrics):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bar_width = 0.35

    r1 = np.arange(len(metrics_names))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, train_metrics, color='b', width=bar_width, edgecolor='grey', label='Train')
    plt.bar(r2, test_metrics, color='g', width=bar_width, edgecolor='grey', label='Test')

    plt.xlabel('Metrics', fontweight='bold')
    plt.xticks(r1 + bar_width / 2, metrics_names)  # Установка меток на оси x
    plt.ylabel('Score', fontweight='bold')
    plt.title('Train and Test Metrics Comparison for KNN')
    plt.legend()
    plt.show()

def main():
    dermatology = fetch_ucirepo(id=33)

    X = dermatology.data.features.dropna()
    y = dermatology.data.targets.values.flatten()
    y = y[X.index]
    y = np.array(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_neighbors': [20, 30, 40, 50],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)

    best_knn = grid.best_estimator_
    best_knn.fit(X_train, y_train)

    y_pred_train = best_knn.predict(X_train)
    y_pred_test = best_knn.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    train_recall = recall_score(y_train, y_pred_train, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    # Визуализация метрик
    train_metrics = [train_accuracy, train_precision, train_recall, train_f1]
    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    plot_metrics(train_metrics, test_metrics)

    # T-SNE
    X_tsne = TSNE(n_components=2, perplexity=20, learning_rate=10, max_iter=1000).fit_transform(X_test)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("KNN -> t-SNE Visualization based on predicted classes")
    plt.show()

    # TriMAP
    trimap_model = trimap.TRIMAP(
        n_dims=2,
        n_inliers=12,
        n_outliers=4,
        n_random=3,
        distance='euclidean',
        lr=0.1,
        n_iters=1000,
        apply_pca=True,
        verbose=True
    )

    X_test_numpy = X_test.to_numpy()
    X_trimap = trimap_model.fit_transform(X_test_numpy)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_trimap[:, 0], X_trimap[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("KNN -> TriMAP Visualization based on predicted classes")
    plt.show()

    # PACMAP
    pacmap_model = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=5,
        MN_ratio=0.5,
        FP_ratio=2.0,
        distance='euclidean',
        lr=1.0,
        num_iters=(100, 100, 250),
        verbose=True,
        apply_pca=True
    )

    X_pacmap = pacmap_model.fit_transform(X_test_numpy)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("KNN -> PACMAP Visualization based on predicted classes")
    plt.show()

    # С истинными метками (для t-SNE, TriMAP и PACMAP)
    for X_embedded, method in zip([X_tsne, X_trimap, X_pacmap], ['t-SNE', 'TriMAP', 'PACMAP']):
        plt.figure(figsize=(10, 5))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='coolwarm', label='True classes')
        plt.colorbar()
        plt.title(f"KNN -> {method} Visualization based on true classes")
        plt.show()

if __name__ == '__main__':
    main()
