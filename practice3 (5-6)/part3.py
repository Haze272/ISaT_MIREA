from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import trimap
import pacmap

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
    plt.title('Train and Test Metrics Comparison for Random Forest')
    plt.legend()
    plt.show()

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

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
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

    # истина
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

    # истина
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

    # истина
    plt.scatter(X_embedded_pacmap[:, 0], X_embedded_pacmap[:, 1], c=y_test, cmap='coolwarm', label='True classes')
    plt.colorbar()
    plt.title("RF -> PaCMAP Visualization based on true classes")
    plt.show()

if __name__ == '__main__':
    main()
