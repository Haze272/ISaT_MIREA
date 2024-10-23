from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

    # Обучить RF-классификатор:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Проверить точность, Recall, Precision и F1:
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred_test, average='weighted'))

    # Перебор параметров с помощью Grid Search:
    param_grid = {
        'n_estimators': [100, 200, 300],  # Количество деревьев в лесу
        'max_depth': [None, 10, 20, 30],  # Глубина деревьев
        'min_samples_split': [2, 5, 10],  # Минимальное число образцов для разделения узла
        'min_samples_leaf': [1, 2, 4],  # Минимальное число образцов в листе
        'bootstrap': [True, False]  # Использовать или нет бутстраппинг
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)

    # Визуализация с использованием t-SNE на основе предсказанных меток:
    X_embedded = TSNE(n_components=2).fit_transform(X_test)

    # Использовать предсказанные метки для раскраски точек
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("t-SNE Visualization based on predicted classes")
    plt.show()

    # Если нужна визуализация на основе настоящих меток:
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("t-SNE Visualization based on true classes")
    plt.show()

if __name__ == '__main__':
    main()