import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def main():
    # 7 điểm dữ liệu theo đề bài: x1..x7
    point_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    X = np.array([
        [1, 1],
        [1, 3],
        [2, 2],
        [4, 4],
        [4, 5],
        [5, 4],
        [5, 5],
    ], dtype=float)

    # Nhập K từ bàn phím (ví dụ nhập 2 hoặc 3)
    k = int(input("Nhập số cụm K (ví dụ 2 hoặc 3): "))
    if k < 1 or k > len(X):
        raise ValueError(f"K phải nằm trong khoảng từ 1 đến {len(X)}.")

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    df_result = pd.DataFrame({
        "Điểm": point_names,
        "X": X[:, 0],
        "Y": X[:, 1],
        "Cụm": labels,
    })
    print(f"\n=== Kết quả K-means với K = {k} ===")
    print(df_result.to_string(index=False))
    print("\nTâm các cụm (Centroids):")
    print(centroids)
    print(f"Tổng bình phương khoảng cách trong cụm (Inertia / SSE): {model.inertia_:.4f}")

    # Trực quan hóa kết quả
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=120, edgecolors="black")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=220, label="Tâm cụm")

    for i, name in enumerate(point_names):
        plt.annotate(f"{name} ({int(X[i, 0])},{int(X[i, 1])})", (X[i, 0] + 0.08, X[i, 1] + 0.08))

    plt.title(f"K-Means Clustering (K = {k})")
    plt.xlabel("Trục X")
    plt.ylabel("Trục Y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


## Bài 2: Gom cụm phân cấp với Single Linkage và Complete Linkage (`bai2_hierarchical.py`)
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


def run_hierarchical(X: np.ndarray, point_names: list[str], method: str, n_clusters: int = 2):
    """Thực hiện gom cụm phân cấp theo phương pháp linkage ('single' hoặc 'complete')."""
    Z = linkage(X, method=method)
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    print(f"\n=== Phân cụm phân cấp - Liên kết: {method.upper()} (tách thành {n_clusters} cụm) ===")
    for name, pt, cl in zip(point_names, X, clusters):
        print(f"  {name} {tuple(pt.astype(int))} -> Cụm {cl}")

    plt.figure(figsize=(8, 4.5))
    dendrogram(Z, labels=point_names)
    plt.title(f"Dendrogram - {method.capitalize()} Linkage")
    plt.xlabel("Điểm dữ liệu")
    plt.ylabel("Khoảng cách")
    plt.tight_layout()
    plt.show()


def main():
    point_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    X = np.array([
        [1, 1],
        [1, 3],
        [2, 2],
        [4, 4],
        [4, 5],
        [5, 4],
        [5, 5],
    ], dtype=float)

    # Liên kết đơn (Single linkage)
    run_hierarchical(X, point_names, method="single", n_clusters=2)

    # Liên kết đầy đủ (Complete linkage)
    run_hierarchical(X, point_names, method="complete", n_clusters=2)


if __name__ == "__main__":
    main()


## Bài 3: K-means kiểm tra hội tụ theo lỗi bình phương với $\delta = 6.2$ (`bai3_kmeans_sse.py`)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_sse(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Tính tổng lỗi bình phương (Sum of Squared Errors - SSE)."""
    sse = 0.0
    for k_idx in range(len(centroids)):
        cluster_points = X[labels == k_idx]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[k_idx]) ** 2)
    return float(sse)


def kmeans_min_sse_threshold(
    X: np.ndarray,
    k: int = 2,
    delta: float = 6.2,
    max_iter: int = 50,
    init_indices: list[int] | None = None,
):
    """Gom cụm K-means và dừng khi tổng lỗi bình phương SSE <= delta."""
    if init_indices is not None and len(init_indices) == k:
        centroids = X[init_indices].astype(float).copy()
    else:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(X), size=k, replace=False)
        centroids = X[chosen].astype(float).copy()

    history_sse = []
    labels = np.zeros(len(X), dtype=int)

    for step in range(1, max_iter + 1):
        # 1. Gán mỗi điểm về tâm cụm gần nhất
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        # 2. Cập nhật tâm cụm
        new_centroids = centroids.copy()
        for k_idx in range(k):
            cluster_points = X[new_labels == k_idx]
            if len(cluster_points) > 0:
                new_centroids[k_idx] = cluster_points.mean(axis=0)

        current_sse = compute_sse(X, new_labels, new_centroids)
        history_sse.append(current_sse)

        print(f"Bước {step}: SSE = {current_sse:.4f} | Ngưỡng delta = {delta}")

        labels = new_labels
        centroids = new_centroids

        if current_sse <= delta:
            print(f"=> Dừng tại bước {step} vì SSE ({current_sse:.4f}) <= delta ({delta}).")
            break
    else:
        print(
            f"=> Đã chạy hết {max_iter} bước nhưng SSE ({history_sse[-1]:.4f}) vẫn > delta ({delta})."
        )

    return labels, centroids, history_sse


def main():
    point_names = ["x1", "x2", "x3", "x4", "x5"]
    # x1=(0, 2), x2=(0, 0), x3=(1.5, 0), x4=(5, 0), x5=(5, 2)
    X = np.array([
        [0.0, 2.0],
        [0.0, 0.0],
        [1.5, 0.0],
        [5.0, 0.0],
        [5.0, 2.0],
    ])
    delta = 6.2
    k = 2

    labels, centroids, history_sse = kmeans_min_sse_threshold(
        X, k=k, delta=delta, init_indices=[0, 3]
    )

    df = pd.DataFrame({
        "Điểm": point_names,
        "X": X[:, 0],
        "Y": X[:, 1],
        "Cụm": labels + 1,
    })
    print("\n--- Bảng kết quả phân cụm ---")
    print(df.to_string(index=False))
    print("\nTâm cụm cuối cùng:")
    print(centroids)

    # Vẽ 2 biểu đồ: Phân cụm và Quá trình giảm SSE so với ngưỡng delta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set1", s=130, edgecolors="black")
    ax1.scatter(centroids[:, 0], centroids[:, 1], c="blue", marker="X", s=220, label="Tâm cụm")
    for i, name in enumerate(point_names):
        ax1.annotate(
            f"{name} ({X[i, 0]:g},{X[i, 1]:g})",
            (X[i, 0] + 0.12, X[i, 1] + 0.08),
        )
    ax1.set_title(f"Bài 3 - Phân cụm K-means (K={k}, delta={delta})")
    ax1.set_xlabel("Trục X")
    ax1.set_ylabel("Trục Y")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    steps = np.arange(1, len(history_sse) + 1)
    ax2.plot(steps, history_sse, marker="o", linewidth=2, label="SSE qua từng bước")
    ax2.axhline(y=delta, color="red", linestyle="--", linewidth=1.8, label=f"Ngưỡng delta = {delta}")
    ax2.set_title("Diễn biến lỗi bình phương SSE")
    ax2.set_xlabel("Bước lặp")
    ax2.set_ylabel("SSE")
    ax2.set_xticks(steps)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()