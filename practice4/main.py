import os
os.environ['OMP_NUM_THREADS'] = '1'

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import trimap
import pacmap


def train_classifiers(X_train, y_train, X_test, y_test):
    classifiers = {
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'RF': RandomForestClassifier()
    }

    results = {}

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[clf_name] = report

    return results


def balance_and_train(X_train, y_train, X_test, y_test):
    resampling_methods = {
        'SMOTE': SMOTE(),
        'Borderline-SMOTE': BorderlineSMOTE(),
        'Borderline-SMOTE2': BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)
    }

    results = {}
    results_x_y = {}

    for method_name, sampler in resampling_methods.items():
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        clf_results = train_classifiers(X_resampled, y_resampled, X_test, y_test)
        results[method_name] = clf_results
        results_x_y[method_name] = [X_resampled, y_resampled]

    return results, results_x_y

def visualize_with_pacmap(X, y, title, ax):
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=0.2)
    embedding = reducer.fit_transform(X.to_numpy())

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=y,
        palette='viridis',
        alpha=0.6,
        ax=ax
    )
    ax.set_title(title)
def visualize_with_trimap(X, y, title, ax):
    reducer = trimap.TRIMAP(n_inliers=2,
                              n_outliers=2,
                              n_random=2)
    embedding = reducer.fit_transform(X.to_numpy())

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=y,
        palette='viridis',
        alpha=0.6,
        ax=ax
    )
    ax.set_title(title)
def visualize_with_tsne(X, y, title, ax):
    reducer = TSNE(n_components=2, perplexity=5, learning_rate=5, max_iter=2500, random_state=42)
    embedding = reducer.fit_transform(X)

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=y,
        palette='viridis',
        alpha=0.6,
        ax=ax
    )
    ax.set_title(title)


def main():
    dermatology = fetch_ucirepo(id=33)

    X = dermatology .data.features.dropna()
    y = dermatology .data.targets.values.flatten()
    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_before = train_classifiers(X_train, y_train, X_test, y_test)
    results_after, results_x_y = balance_and_train(X_train, y_train, X_test, y_test)

    print("Until balancing:")
    for clf_name, metrics in results_before.items():
        print(f"\nClassificator: {clf_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Recall: {metrics['1']['recall']:.4f}")
        print(f"F1-score: {metrics['1']['f1-score']:.4f}")

    print("\nAfter balancing:")
    for method_name, classifiers in results_after.items():
        print(f"\nMethod: {method_name}")
        for clf_name, metrics in classifiers.items():
            print(f"\nClassificator: {clf_name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Recall: {metrics['1']['recall']:.4f}")
            print(f"F1-score: {metrics['1']['f1-score']:.4f}")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    visualize_with_pacmap(X, y, 'Source data', ax1)

    X_resampled_smote = results_x_y['SMOTE'][0]
    y_resampled_smote = results_x_y['SMOTE'][1]
    visualize_with_pacmap(X_resampled_smote, y_resampled_smote, 'After balancing (SMOTE)', ax2)

    X_resampled_bsmote = results_x_y['Borderline-SMOTE'][0]
    y_resampled_bsmote = results_x_y['Borderline-SMOTE'][1]
    visualize_with_pacmap(X_resampled_bsmote, y_resampled_bsmote, 'After balancing (Borderline-SMOTE)', ax3)

    X_resampled_bsmote2 = results_x_y['Borderline-SMOTE2'][0]
    y_resampled_bsmote2 = results_x_y['Borderline-SMOTE2'][1]
    visualize_with_pacmap(X_resampled_bsmote2, y_resampled_bsmote2, 'After balancing (Borderline-SMOTE2)', ax4)

    plt.tight_layout()
    plt.suptitle('visualize_with_pacmap')
    plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    visualize_with_trimap(X, y, 'Source data', ax1)

    X_resampled_smote = results_x_y['SMOTE'][0]
    y_resampled_smote = results_x_y['SMOTE'][1]
    visualize_with_trimap(X_resampled_smote, y_resampled_smote, 'After balancing (SMOTE)', ax2)

    X_resampled_bsmote = results_x_y['Borderline-SMOTE'][0]
    y_resampled_bsmote = results_x_y['Borderline-SMOTE'][1]
    visualize_with_trimap(X_resampled_bsmote, y_resampled_bsmote, 'After balancing (Borderline-SMOTE)', ax3)

    X_resampled_bsmote2 = results_x_y['Borderline-SMOTE2'][0]
    y_resampled_bsmote2 = results_x_y['Borderline-SMOTE2'][1]
    visualize_with_trimap(X_resampled_bsmote2, y_resampled_bsmote2, 'After balancing (Borderline-SMOTE2)', ax4)

    plt.tight_layout()
    plt.suptitle('visualize_with_trimap')
    plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    visualize_with_tsne(X, y, 'Source data', ax1)

    X_resampled_smote = results_x_y['SMOTE'][0]
    y_resampled_smote = results_x_y['SMOTE'][1]
    visualize_with_tsne(X_resampled_smote, y_resampled_smote, 'After balancing (SMOTE)', ax2)

    X_resampled_bsmote = results_x_y['Borderline-SMOTE'][0]
    y_resampled_bsmote = results_x_y['Borderline-SMOTE'][1]
    visualize_with_tsne(X_resampled_bsmote, y_resampled_bsmote, 'After balancing (Borderline-SMOTE)', ax3)

    X_resampled_bsmote2 = results_x_y['Borderline-SMOTE2'][0]
    y_resampled_bsmote2 = results_x_y['Borderline-SMOTE2'][1]
    visualize_with_tsne(X_resampled_bsmote2, y_resampled_bsmote2, 'After balancing (Borderline-SMOTE2)', ax4)

    plt.tight_layout()
    plt.suptitle('visualize_with_tsne')
    plt.show()


if __name__ == '__main__':
    main()