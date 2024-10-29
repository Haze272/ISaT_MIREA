import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import umap
import trimap
import pacmap

# Шаг 1: Загрузка данных и предобработка
# Загружаем данные из файла, подгоняем под формат
columns = ['Country', 'Landmass', 'Zone', 'Area', 'Population', 'Language', 'Religion',
           'Bars', 'Stripes', 'Colours', 'Red', 'Green', 'Blue', 'Gold', 'White',
           'Black', 'Orange', 'Mainhue', 'Circles', 'Crosses', 'Saltires', 'Quarters',
           'Sunstars', 'Crescent', 'Triangle', 'Icon', 'Animate', 'Text', 'Topleft', 'Botright']
data = pd.read_csv('flag.data', header=None, names=columns)

# Убираем нечисловые столбцы (например, 'Country' и 'Mainhue', 'Topleft', 'Botright')
non_numeric_columns = ['Country', 'Mainhue', 'Topleft', 'Botright']
data_numeric = data.drop(non_numeric_columns, axis=1)

# Преобразуем категориальные данные в числовые с помощью LabelEncoder
label_encoders = {}
for column in non_numeric_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Шаг 2: Применение алгоритмов снижения размерности

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_numeric)

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
data_umap = umap_model.fit_transform(data_numeric)

# TriMAP
trimap_model = trimap.TRIMAP()
data_trimap = trimap_model.fit_transform(data_numeric.values)

# PacMAP
pacmap_model = pacmap.PaCMAP(n_components=2, random_state=42)
data_pacmap = pacmap_model.fit_transform(data_numeric)

# Шаг 3: Визуализация результатов
plt.figure(figsize=(16, 12))

# t-SNE
plt.subplot(2, 2, 1)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data['Landmass'], cmap='Spectral', s=50)
plt.title('t-SNE')

# UMAP
plt.subplot(2, 2, 2)
plt.scatter(data_umap[:, 0], data_umap[:, 1], c=data['Landmass'], cmap='Spectral', s=50)
plt.title('UMAP')

# TriMAP
plt.subplot(2, 2, 3)
plt.scatter(data_trimap[:, 0], data_trimap[:, 1], c=data['Landmass'], cmap='Spectral', s=50)
plt.title('TriMAP')

# PacMAP
plt.subplot(2, 2, 4)
plt.scatter(data_pacmap[:, 0], data_pacmap[:, 1], c=data['Landmass'], cmap='Spectral', s=50)
plt.title('PacMAP')

plt.show()
