
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import trimap
import os
os.environ['OMP_NUM_THREADS'] = '1'


def visualize_with_trimap(scaled_data, title):
    embedding = trimap.TRIMAP().fit_transform(scaled_data)
    embedding = trimap.TRIMAP(knn_tuple=(5, 5)).fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("trimap component 1")
    plt.ylabel("trimap component 2")
    plt.show()

def main():
    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice2/mammoth2.csv")
    numerical_data = df.select_dtypes(include=['float64', 'int64'])
    numerical_data = numerical_data.sample(frac=0.1, random_state=42)

    scalers = {
        'MinMax Scaling': MinMaxScaler(),
        'Standard Scaling': StandardScaler(),
        'Robust Scaling': RobustScaler()
    }
    for scaler_name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(df)
        visualize_with_trimap(scaled_data, f"trimap with {scaler_name}")


if __name__ == '__main__':
    main()