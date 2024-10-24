from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import trimap
import pacmap
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Загрузка набора данных
    dermatology = fetch_ucirepo(id=33)

    X = dermatology.data.features.dropna()
    y = dermatology.data.targets.values.flatten()
    y = y[X.index]
    y = np.array(y).astype(int)  # Преобразуем целевые значения в целые числа

    # Разделить данные на обучающую и тестовую выборки:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Перебор параметров с помощью Grid Search:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)

    # Обучение SVM-классификатора с лучшими параметрами:
    best_svm = grid.best_estimator_
    best_svm.fit(X_train, y_train)

    # Проверка точности, Recall, Precision и F1:
    y_pred_train = best_svm.predict(X_train)
    y_pred_test = best_svm.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred_test, average='weighted'))

    # Число опорных векторов:
    print("Number of support vectors:", best_svm.n_support_)

    # Применение t-SNE к тестовым данным:
    X_tsne = TSNE(n_components=2).fit_transform(X_test)

    # Визуализация с t-SNE
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("SVM -> t-SNE Visualization based on predicted classes")
    plt.show()

    # Визуализация с истинными метками
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("SVM -> t-SNE Visualization based on true classes")
    plt.show()

    # TriMAP
    X_test_np = X_test.to_numpy()  # Преобразуем DataFrame в NumPy массив
    trimap_model = trimap.TRIMAP(n_dims=2, n_inliers=12, n_outliers=4, n_random=3, distance='euclidean', lr=0.1, n_iters=1000, apply_pca=True)
    X_trimap = trimap_model.fit_transform(X_test_np)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_trimap[:, 0], X_trimap[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("SVM -> TriMAP Visualization based on predicted classes")
    plt.show()

    # Визуализация TriMAP с истинными метками
    plt.figure(figsize=(10, 5))
    plt.scatter(X_trimap[:, 0], X_trimap[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("SVM -> TriMAP Visualization based on true classes")
    plt.show()

    # PACMAP
    pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0,
                                  distance='euclidean', lr=1.0, num_iters=(100, 100, 250), verbose=True,
                                  apply_pca=True)
    X_pacmap = pacmap_model.fit_transform(X_test_np)  # Используем NumPy массив

    plt.figure(figsize=(10, 5))
    plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("SVM -> PACMAP Visualization based on predicted classes")
    plt.show()

    # Визуализация PACMAP с истинными метками
    plt.figure(figsize=(10, 5))
    plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("SVM -> PACMAP Visualization based on true classes")
    plt.show()

if __name__ == '__main__':
    main()
