import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pacmap
import matplotlib.pyplot as plt
import os
os.environ['OMP_NUM_THREADS'] = '1'

def visualize_with_pacmap(scaled_data, title):
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)

    pacmac_result = embedding.fit_transform(scaled_data, init="pca")

    plt.figure(figsize=(8, 6))
    plt.scatter(pacmac_result[:, 0], pacmac_result[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("pacmap component 1")
    plt.ylabel("pacmap component 2")
    plt.show()
def main():
    # Загрузка данных
    df = pd.read_csv("FILEPATH")

    # Убираем ненужные колонки, если есть (например, текстовые данные)
    numerical_data = df.select_dtypes(include=['float64', 'int64'])

    # Пример уменьшения данных до 10%
    numerical_data = numerical_data.sample(frac=0.1, random_state=42)

    # Масштабирование данных
    scalers = {
        'MinMax Scaling': MinMaxScaler(),
        'Standard Scaling': StandardScaler(),
        'Robust Scaling': RobustScaler()
    }

    # Применяем t-SNE и визуализируем для каждого метода масштабирования
    for scaler_name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(df)
        visualize_with_pacmap(scaled_data, f"pacmap with {scaler_name}")


if __name__ == '__main__':
    main()