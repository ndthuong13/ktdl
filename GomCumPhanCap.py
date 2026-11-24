"""
Bài 9 - Thực hành: Gom cụm phân cấp (Hierarchical Clustering) trên bộ dữ liệu Wine
(chạy trên PyCharm + Anaconda)

Cài thư viện (nếu thiếu), mở Anaconda Prompt:
    conda install numpy scipy pandas matplotlib seaborn scikit-learn

Dữ liệu: tải từ UCI; nếu không có mạng sẽ tự dùng sklearn.datasets.load_wine
(cùng dữ liệu, cùng thứ tự cột).
"""
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

NAMES = ['Type', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity', 'Magnesium',
         'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
         'Proanthocyanins', 'Color_intensity', 'Hue',
         'OD280/OD315_of_diluted_wines', 'Proline']
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"


# ---------------------------------------------------------------------------
# 7. Đọc và khảo sát dữ liệu
# ---------------------------------------------------------------------------
def load_wine_data():
    try:
        data = pd.read_csv(URL, names=NAMES, header=None)
    except Exception as e:  # không có mạng / link UCI đổi
        print('Không tải được từ UCI, dùng sklearn.load_wine thay thế:', e)
        from sklearn.datasets import load_wine
        w = load_wine()
        data = pd.DataFrame(w.data, columns=NAMES[1:])
        data.insert(0, 'Type', w.target + 1)  # nhãn 0-2 -> 1-3
    return data


def explore(data):
    print(data.shape)
    print(data.head(100))
    print(data.describe())
    print(data['Type'].value_counts())
    print('Có giá trị thiếu?', data.isnull().values.any())
    data.info()

    data[['Type']].hist()
    plt.show()

    wine_X = data.iloc[:, 1:].values
    wine_Y = data['Type'].values
    plt.figure(figsize=(10, 5))
    # Bài gốc dùng nhãn 0,1,2 nên "Type 0" bị rỗng; nhãn thật là 1,2,3
    for t, c in zip([1, 2, 3], ['orange', 'yellow', 'green']):
        plt.scatter(wine_X[wine_Y == t, 0], wine_X[wine_Y == t, 12],
                    s=80, c=c, label=f'Type {t}')
    plt.xlabel('Alcohol')
    plt.ylabel('Proline')
    plt.title("Wine Dataset Plot")
    plt.legend()
    plt.show()

    # Bài gốc dùng iloc[:, [11, 13]] (Hue, Proline) nhưng đặt nhãn là
    # Color_intensity / OD280 -> sửa lại cho khớp tên cột
    X2 = data[['Color_intensity', 'OD280/OD315_of_diluted_wines']].values
    plt.figure(figsize=(12, 8))
    plt.scatter(X2[:, 0], X2[:, 1], lw=0, s=40)
    plt.xlabel('Color_intensity')
    plt.ylabel('OD280/OD315_of_diluted_wines')
    plt.title('Distribution of Wine Dataset')
    plt.show()


# ---------------------------------------------------------------------------
# 8. Chuẩn hóa + giảm chiều
# ---------------------------------------------------------------------------
def preprocess(data):
    y = data['Type']
    datax = data.drop(['Type'], axis=1)
    print(datax.shape, y.shape)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        datax, y, test_size=0.2, random_state=0)
    print(Xtrain.shape, Xtest.shape)

    std = MinMaxScaler()
    X_std = std.fit_transform(Xtrain)

    X_tsne = TSNE(n_components=2, random_state=123).fit_transform(X_std)
    pca = PCA(n_components=2)
    Xt = pca.fit_transform(X_std)
    print('PCA shape:', Xt.shape)
    return std, X_std, X_tsne, Xtest, ytrain, ytest


# ---------------------------------------------------------------------------
# 9. Dendrogram
# ---------------------------------------------------------------------------
def plot_dendrogram(X_std):
    plt.figure(figsize=(20, 7))
    plt.title("Wine Dendograms")
    shc.dendrogram(shc.linkage(X_std, method='ward'))
    plt.axhline(6, linestyle='--')
    plt.xlabel('sample indice')
    plt.ylabel('dissimilarity metric cluster')
    plt.show()
    # Ngưỡng t = 6 -> 2 cụm; t = 5 -> 3 cụm (theo nhận xét bài gốc)


# ---------------------------------------------------------------------------
# 10. Gom cụm, đánh giá, trực quan hóa
# ---------------------------------------------------------------------------
def match_clusters_to_classes(y_true, labels):
    """Nhãn cụm (0,1,2) là tùy ý nên cần ghép với nhãn thật trước khi tính accuracy.
    Dùng thuật toán Hungarian thay cho hàm converty() gán cứng 1->2, 2->1, 3->0."""
    classes = np.unique(y_true)
    clusters = np.unique(labels)
    cm = np.zeros((len(clusters), len(classes)), dtype=int)
    for i, c in enumerate(clusters):
        for j, k in enumerate(classes):
            cm[i, j] = np.sum((labels == c) & (y_true == k))
    r, c = linear_sum_assignment(-cm)
    mapping = {clusters[i]: classes[j] for i, j in zip(r, c)}
    return np.array([mapping[l] for l in labels])


def _plot_scatter(X, labels, title='t-sne visualization'):
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, c=palette[labels.astype(int)])

    for i in range(num_classes):
        xtext, ytext = np.median(X[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
    plt.title(title)
    plt.show()


def cluster_and_evaluate(std, X_std, X_tsne, Xtest, ytrain, ytest):
    # affinity='euclidean' bị gỡ ở sklearn >= 1.4; ward mặc định dùng euclidean
    cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = cluster.fit_predict(X_std)

    yt = ytrain.to_numpy()
    labels_mapped = match_clusters_to_classes(yt, labels)
    print('Accuracy (train):', round(accuracy_score(yt, labels_mapped) * 100, 4))

    # AgglomerativeClustering không có predict() nên gom cụm lại trên tập test
    X_test = std.transform(Xtest)
    label_test = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_test)
    y_test = ytest.to_numpy()
    label_test_mapped = match_clusters_to_classes(y_test, label_test)
    print('Accuracy (test):', round(accuracy_score(y_test, label_test_mapped) * 100, 4))

    _plot_scatter(X_tsne, labels)


if __name__ == '__main__':
    wine = load_wine_data()
    explore(wine)
    std_, X_std_, X_tsne_, Xtest_, ytrain_, ytest_ = preprocess(wine)
    plot_dendrogram(X_std_)
    cluster_and_evaluate(std_, X_std_, X_tsne_, Xtest_, ytrain_, ytest_)