"""
Bài 9-1: K-Means Clustering (chạy trên PyCharm + Anaconda)

Cài thư viện (nếu thiếu), mở Anaconda Prompt:
    conda install numpy scipy pandas matplotlib seaborn scikit-learn

Dữ liệu: đặt file Mall_Customers.csv cùng thư mục với file .py này
(chỉ cần cho phần 4 - Elbow).
"""
import time
from pathlib import Path

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

BASE_DIR = Path(__file__).resolve().parent
np.random.seed(11)

# ---------------------------------------------------------------------------
# 3. Ví dụ minh họa: tự cài đặt thuật toán K-Means
# ---------------------------------------------------------------------------
def kmeans_display(X, label, title=""):
    """Vẽ dữ liệu theo nhãn (hỗ trợ tối đa 3 cụm như bài gốc)."""
    styles = [('b^', 0), ('go', 1), ('rs', 2)]
    for fmt, k in styles:
        Xk = X[label == k, :]
        plt.plot(Xk[:, 0], Xk[:, 1], fmt, markersize=4, alpha=.8)
    plt.axis('equal')
    plt.title(title)
    plt.show()


def kmeans_init_centers(X, k):
    # chọn ngẫu nhiên k hàng của X làm tâm ban đầu
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(X, centers):
    # khoảng cách giữa mọi điểm và các tâm, trả về chỉ số tâm gần nhất
    D = cdist(X, centers)
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    # True nếu hai tập tâm giống nhau
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return centers, labels, it


def section_3_kmeans_from_scratch():
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)
    X = np.concatenate((X0, X1, X2), axis=0)
    K = 3
    original_label = np.asarray([0] * N + [1] * N + [2] * N).T

    kmeans_display(X, original_label, "Du lieu voi nhan ban dau")

    centers, labels, it = kmeans(X, K)
    print('Số vòng lặp:', it)
    print('Centers found by our algorithm:')
    print(centers[-1])
    kmeans_display(X, labels[-1], "K-Means tu cai dat")

    # So sánh với scikit-learn
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    print('Centers found by scikit-learn:')
    print(km.cluster_centers_)
    kmeans_display(X, km.predict(X), "K-Means scikit-learn")


# ---------------------------------------------------------------------------
# 4. Phương pháp Elbow (dữ liệu Mall_Customers.csv)
# ---------------------------------------------------------------------------
def section_4_elbow():
    csv_path = BASE_DIR / 'Mall_Customers.csv'
    if not csv_path.exists():
        print(f"[Bỏ qua phần 4] Không thấy file {csv_path}. "
              f"Hãy đặt Mall_Customers.csv cùng thư mục với script.")
        return

    dataset = pd.read_csv(csv_path)
    # cột 3, 4 = Annual Income (k$), Spending Score (1-100)
    X = dataset.iloc[:, [3, 4]].values

    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.xlabel('number of clusters')
    plt.ylabel('wcss')
    plt.title('Elbow method')
    plt.show()

    # Khuỷu tay tại K = 5
    km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    y_kmeans = km.fit_predict(X)

    colors = ['red', 'blue', 'green', 'violet', 'yellow']
    for k in range(5):
        plt.scatter(X[y_kmeans == k, 0], X[y_kmeans == k, 1],
                    s=60, c=colors[k], label=f'Cluster {k + 1}')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=100, c='black', label='Centroids')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Biểu diễn dữ liệu đa chiều bằng t-SNE
# ---------------------------------------------------------------------------
def _plot_kmean_scatter(X, labels):
    """X: dữ liệu 2 chiều sau giảm chiều; labels: nhãn dự báo."""
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    ax.scatter(X[:, 0], X[:, 1], lw=0, s=40,
               c=palette[labels.astype(int)])  # np.int đã bị gỡ -> dùng int
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    for i in range(num_classes):
        xtext, ytext = np.median(X[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
    plt.title('t-sne visualization')
    plt.show()


def section_5_tsne():
    X, _ = make_blobs(n_samples=250, centers=2, n_features=5,
                      cluster_std=3.0, random_state=123)

    kmean = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmean.fit(X)
    labels = kmean.predict(X)

    time_start = time.time()
    X_tsne = TSNE(n_components=2, random_state=123).fit_transform(X)
    print('t-SNE done! Time elapsed: {:.3f} seconds'.format(time.time() - time_start))
    print(X_tsne.shape)

    _plot_kmean_scatter(X_tsne, labels)


if __name__ == '__main__':
    section_3_kmeans_from_scratch()
    section_4_elbow()
    section_5_tsne()