from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import trimap
import pacmap


def main():
    dermatology = fetch_ucirepo(id=33)

    X = dermatology.data.features.dropna()
    y = dermatology.data.targets.values.flatten()
    y = y[X.index]
    y = np.array(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_np = X_test.to_numpy()

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred_test, average='weighted'))

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

    # T-SNE:
    X_embedded_tsne = TSNE(n_components=2).fit_transform(X_test_np)
    plt.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=y_pred_test, cmap='viridis', label='Predicted classes')
    plt.colorbar()
    plt.title("RF -> t-SNE Visualization based on predicted classes")
    plt.show()

    # истиные
    plt.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("RF -> t-SNE Visualization based on true classes")
    plt.show()

    # TRIMAP
    trimap_model = trimap.TRIMAP()
    X_embedded_trimap = trimap_model.fit_transform(X_test_np)
    plt.scatter(X_embedded_trimap[:, 0], X_embedded_trimap[:, 1], c=y_pred_test, cmap='viridis',
                label='Predicted classes')
    plt.colorbar()
    plt.title("RF -> TRIMAP Visualization based on predicted classes")
    plt.show()

    # истиные
    plt.scatter(X_embedded_trimap[:, 0], X_embedded_trimap[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("RF -> TRIMAP Visualization based on true classes")
    plt.show()

    # PaCMAP
    pacmap_model = pacmap.PaCMAP(n_neighbors=5, random_state=42)
    X_embedded_pacmap = pacmap_model.fit_transform(X_test_np)
    plt.scatter(X_embedded_pacmap[:, 0], X_embedded_pacmap[:, 1], c=y_pred_test, cmap='viridis',
                label='Predicted classes')
    plt.colorbar()
    plt.title("RF -> PaCMAP Visualization based on predicted classes")
    plt.show()

    # истиные
    plt.scatter(X_embedded_pacmap[:, 0], X_embedded_pacmap[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("RF -> PaCMAP Visualization based on true classes")
    plt.show()


if __name__ == '__main__':
    main()
