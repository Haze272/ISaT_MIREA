import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_with_umap(scaled_data, title):
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')
    embedding = umap_model.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("umap component 1")
    plt.ylabel("umap component 2")
    plt.show()
def main():
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice2 (3-4)/mammoth2.csv")

    numerical_data = df.select_dtypes(include=['float64', 'int64'])
    numerical_data = numerical_data.sample(frac=0.1, random_state=42)
    scalers = {
        'MinMax Scaling': MinMaxScaler(),
        'Standard Scaling': StandardScaler(),
        'Robust Scaling': RobustScaler()
    }

    for scaler_name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(numerical_data)
        visualize_with_umap(scaled_data, f"umap with {scaler_name}")

if __name__ == '__main__':
    main()
    print('loh')