import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
os.environ['OMP_NUM_THREADS'] = '1'

def visualize_with_tsne(scaled_data, title):
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, max_iter=500, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.show()


def main():
    df = pd.read_csv("FILEPATH")
    numerical_data = df.select_dtypes(include=['float64', 'int64'])
    numerical_data = numerical_data.sample(frac=0.1, random_state=42)

    scalers = {
        'MinMax Scaling': MinMaxScaler(),
        'Standard Scaling': StandardScaler(),
        'Robust Scaling': RobustScaler()
    }

    for scaler_name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(numerical_data)
        visualize_with_tsne(scaled_data, f"t-SNE with {scaler_name}")


if __name__ == '__main__':
    main()
    print('end')