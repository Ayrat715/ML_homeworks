import numpy as np # для работы с массивами и мат операциями
import matplotlib.pyplot as plt # для построения графиков
from sklearn.cluster import KMeans # для кластеризации с алгоритмом k-means из библиотеки scikit-learn
from sklearn.datasets import load_iris # для загрузки  датасета
import imageio # для создания GIF из полученных графиков
import os # для работы с файловой системой

iris = load_iris()
X = iris.data # загрузка и сохранение датасета в массиве X.


# поиск оптимального количества кластеров с использованием метода локтя
wcss = [] # Within-Cluster Sum of Squares // сумма квадратов расстояний от каждой точки до центроида её кластера
for i in range(1, 11):
    '''
    :param n_clusters — количество кластеров
    :param init — инициализация центроидов
    :param max_iter — максимальное количество итераций
    :param n_init — число запусков алгоритма с разными начальными условиями
    :param random_state — воспроизводимость результатов
    '''
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# На графике по оси X откладывается количество кластеров, а по оси Y – WCSS
# Метод локтя помогает определить оптимальное число кластеров, где происходит резкое снижение WCSS
plt.plot(range(1, 11), wcss)
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()


# реализация собственного алгоритма k-means
# функция для инициализации центроидов
'''
np.random.choice выбирает k уникальных индексов из диапазона количества образцов (X.shape[0]) 
(из датасета в качестве начальных центроидов)
'''
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# функция для присвоения точек кластерам
'''
для каждой точки x вычисляются расстояния до всех центроидов с помощью нормы
np.argmin(distances) возвращает индекс ближайшего центроида, и этот индекс записывается как номер кластера для точки
'''
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = np.linalg.norm(x - centroids, axis=1)
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# функция для обновления центроидов
# для каждого кластера вычисляется новый центроид как среднее значение всех точек, принадлежащих этому кластеру
def update_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[clusters == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids
# функция для реализации итераций k-means
'''
функция реализована как генератор, который на каждой итерации:
1. присваивает точки кластерам (используя текущие центроиды)
2. вычисляет новые центроиды
3. если центроиды не изменились (условие сходимости), цикл прерывается
4. yield возвращает текущие центроиды и распределение точек по кластерам для визуализации каждого шага
'''
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        yield centroids, clusters


# визуализация шагов алгоритма и создание анимации

# определение оптимального числа кластеров
optimal_k = 3

# создание папки для изображений, если отсутствует
os.makedirs('kmeans_images', exist_ok=True)

'''
для каждой итерации работы алгоритма (каждое обновление центроидов) создаётся график
для каждого кластера точки, принадлежащие кластеру, отображаются разными цветами
центроиды выделены красными крестиками (marker='X')
график сохраняется как изображение, которое потом добавляется в список images
'''
images = []
for i, (centroids, clusters) in enumerate(kmeans(X, optimal_k)):
    plt.figure()
    for cluster in range(optimal_k):
        plt.scatter(X[clusters == cluster, 0], X[clusters == cluster, 1], label=f'Cluster {cluster + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')
    plt.legend()
    plt.title(f'Iteration {i + 1}')
    file_path = f'kmeans_images/step_{i + 1}.png'
    plt.savefig(file_path)

    images.append(imageio.imread(file_path))
    plt.close()

# создаётся анимация из последовательности изображений, где каждая картинка соответствует одной итерации алгоритма
imageio.mimsave('kmeans_animation.gif', images, duration=1)