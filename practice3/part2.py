from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
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
        'n_neighbors': [5, 7, 9, 11, 15, 20],  # Увеличение диапазона соседей
        'weights': ['uniform', 'distance'],  # Использовать взвешивание по расстоянию
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)

    # Обучить KNN-классификатор с лучшими параметрами:
    best_knn = grid.best_estimator_
    best_knn.fit(X_train, y_train)

    # Проверить точность, Recall, Precision и F1:
    y_pred_train = best_knn.predict(X_train)
    y_pred_test = best_knn.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred_test, average='weighted'))

    # Применение t-SNE к тестовым данным:
    X_embedded = TSNE(n_components=2, perplexity=50, learning_rate=100).fit_transform(X_test)

    # Визуализация с предсказанными метками:
    plt.figure(figsize=(10, 5))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("KNN -> t-SNE Visualization based on predicted classes")
    plt.show()

    # Визуализация с истинными метками:
    plt.figure(figsize=(10, 5))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("KNN -> t-SNE Visualization based on true classes")
    plt.show()

if __name__ == '__main__':
    main()