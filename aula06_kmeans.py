import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

# Carregar uma imagem de exemplo (neste caso, uma imagem de flores)
image = load_sample_image("flower.jpg")

# Converte a imagem em uma matriz numérica (Altura x Largura x 3 canais de cor: vermelho, verde, azul)
image = np.array(image, dtype=np.float64) / 255  # Normaliza os valores de pixel para o intervalo [0, 1]

# Transforma a matriz da imagem em um vetor de pixels (Altura*Largura x 3)
w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, d))

# Aplica o K-means para agrupar os pixels
n_clusters = 8  # Número de clusters (cores finais)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(image_array)
labels = kmeans.predict(image_array)
centers = kmeans.cluster_centers_

# Atribui a cor média do cluster a cada pixel
image_recolored = centers[labels].reshape(image.shape)

# Mostra a imagem original e a imagem recolorida
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_recolored)
plt.title("Imagem Recolorida (K-means)")
plt.axis('off')

plt.show()
