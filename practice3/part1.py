from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
# import umap
import matplotlib.pyplot as plt
import numpy as np




def main():
    dermatology = fetch_ucirepo(id=33)

    X = dermatology.data.features.dropna()
    y = dermatology.data.targets.values.flatten()
    y = y[X.index]
    y = np.array(y).astype(int)  # Преобразуем целевые значения в целые числа

    # Разделить данные на обучающую и тестовую выборки:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучить SVM-классификатор:
    clf = SVC(kernel='linear', C=1)  # Начнем с линейного ядра
    clf.fit(X_train, y_train)

    # Проверить точность, Recall, Precision и F1:
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred_test, average='weighted'))

    # Число опорных векторов:
    print("Number of support vectors:", clf.n_support_)

    # Перебор параметров с помощью Grid Search:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)

    # Визуализация с использованием t-SNE, UMAP, и других алгоритмов: Для этого можно использовать следующие библиотеки:
    # Пример для t-SNE
    X_embedded = TSNE(n_components=2).fit_transform(X_test)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test)
    plt.title("t-SNE Visualization")
    plt.show()

if __name__ == '__main__':
    main()