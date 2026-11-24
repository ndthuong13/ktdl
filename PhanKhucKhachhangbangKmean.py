"""
Bài 9-2: Phân khúc khách hàng bằng K-Means (chạy trên PyCharm + Anaconda)

Cài thư viện (nếu thiếu), mở Anaconda Prompt:
    conda install numpy pandas matplotlib seaborn scikit-learn plotly

Dữ liệu: đặt file Mall_Customers.csv cùng thư mục với file .py này.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'Mall_Customers.csv'

AGE = 'Age'
INCOME = 'Annual Income (k$)'
SCORE = 'Spending Score (1-100)'
FEATURES = [AGE, INCOME, SCORE]


# ---------------------------------------------------------------------------
# 1. Khám phá dữ liệu
# ---------------------------------------------------------------------------
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy {CSV_PATH}. Hãy tải Mall_Customers.csv và đặt cùng thư mục với script.")
    df = pd.read_csv(CSV_PATH)
    print(df.head())
    print('shape:', df.shape)
    print(df.describe())
    print(df.dtypes)
    print(df.isnull().sum())
    return df


# ---------------------------------------------------------------------------
# 2. Trực quan hóa dữ liệu
# ---------------------------------------------------------------------------
def visualize(df):
    plt.style.use('fivethirtyeight')

    # Histograms (sns.distplot đã bị loại bỏ -> dùng histplot + kde)
    plt.figure(figsize=(15, 6))
    for n, x in enumerate(FEATURES, start=1):
        plt.subplot(1, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[x], bins=20, kde=True, stat='density')
        plt.title('Distplot of {}'.format(x))
    plt.show()

    # Count plot theo giới tính
    plt.figure(figsize=(15, 5))
    sns.countplot(y='Gender', data=df)
    plt.show()

    # Quan hệ giữa các biến
    plt.figure(figsize=(15, 7))
    n = 0
    for x in FEATURES:
        for y in FEATURES:
            n += 1
            plt.subplot(3, 3, n)
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            sns.regplot(x=x, y=y, data=df)
            plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
    plt.show()

    # Scatter theo giới tính
    for xcol, ycol in [(AGE, INCOME), (INCOME, SCORE)]:
        plt.figure(figsize=(15, 6))
        for gender in ['Male', 'Female']:
            sub = df[df['Gender'] == gender]
            plt.scatter(sub[xcol], sub[ycol], s=200, alpha=0.5, label=gender)
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title('{} vs {} w.r.t Gender'.format(xcol, ycol))
        plt.legend()
        plt.show()

    # Boxplot
    plt.figure(figsize=(15, 7))
    for n, cols in enumerate(FEATURES, start=1):
        plt.subplot(1, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.boxplot(x=cols, y='Gender', data=df, hue='Gender', legend=False)
        plt.ylabel('Gender' if n == 1 else '')
        plt.title('Boxplots' if n == 2 else '')
    plt.show()

    # Violin + swarm
    plt.figure(figsize=(15, 7))
    for n, cols in enumerate(FEATURES, start=1):
        plt.subplot(1, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.violinplot(x=cols, y='Gender', data=df, hue='Gender',
                       palette='vlag', legend=False)
        sns.swarmplot(x=cols, y='Gender', data=df, color='k', size=3)
        plt.ylabel('Gender' if n == 1 else '')
        plt.title('Violinplots & Swarmplots' if n == 2 else '')
    plt.show()


# ---------------------------------------------------------------------------
# 3. Phân cụm K-Means
# ---------------------------------------------------------------------------
def make_kmeans(n):
    return KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                  tol=0.0001, random_state=111, algorithm='elkan')


def plot_inertia(X, title):
    inertia = []
    for n in range(1, 11):
        algo = make_kmeans(n)
        algo.fit(X)
        inertia.append(algo.inertia_)
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(1, 11), inertia, 'o')
    plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title(title)
    plt.show()


def plot_2d_clusters(df, X, n_clusters, xcol, ycol, h=0.1):
    """Huấn luyện K-Means và vẽ vùng biên giữa các cụm.
    h=0.1 (bài gốc dùng 0.02) để chạy nhanh và đỡ tốn RAM."""
    algo = make_kmeans(n_clusters)
    algo.fit(X)
    labels = algo.labels_
    centroids = algo.cluster_centers_

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = algo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(15, 7))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
    plt.scatter(df[xcol], df[ycol], c=labels, s=200)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', alpha=0.5)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.show()
    return labels


def segmentation_age_score(df):
    X1 = df[[AGE, SCORE]].values
    plot_inertia(X1, 'Age & Spending Score - Inertia')
    plot_2d_clusters(df, X1, 4, AGE, SCORE)


def segmentation_income_score(df):
    X2 = df[[INCOME, SCORE]].values
    plot_inertia(X2, 'Annual Income & Spending Score - Inertia')
    plot_2d_clusters(df, X2, 5, INCOME, SCORE)


def segmentation_3d(df):
    X3 = df[FEATURES].values
    plot_inertia(X3, 'Age, Income & Spending Score - Inertia')

    algo = make_kmeans(6)
    algo.fit(X3)
    df['label3'] = algo.labels_

    trace1 = go.Scatter3d(
        x=df[AGE], y=df[SCORE], z=df[INCOME],
        mode='markers',
        marker=dict(color=df['label3'], size=8, opacity=0.8))
    layout = go.Layout(
        title='Clusters',
        scene=dict(xaxis=dict(title='Age'),
                   yaxis=dict(title='Spending Score'),
                   zaxis=dict(title='Annual Income')))
    fig = go.Figure(data=[trace1], layout=layout)
    # Mở trong trình duyệt (thay cho py.offline.iplot chỉ dùng được trong notebook)
    fig.show(renderer='browser')
    # Hoặc lưu ra file HTML:
    fig.write_html(str(BASE_DIR / 'clusters_3d.html'))
    print('Đã lưu clusters_3d.html')


if __name__ == '__main__':
    data = load_data()
    visualize(data)
    segmentation_age_score(data)
    segmentation_income_score(data)
    segmentation_3d(data)