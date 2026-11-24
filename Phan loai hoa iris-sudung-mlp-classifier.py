#pip install numpy pandas scikit-learn tensorflow matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from sklearn import datasets, metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.utils import to_categorical


def evaluate_model(y_true, y_pred, target_names, title: str):
    """In báo cáo phân lớp, ma trận nhầm lẫn và vẽ biểu đồ ma trận nhầm lẫn."""
    print(f"\n==================== {title} ====================")
    print(f"Accuracy: {metrics.accuracy_score(y_true, y_pred):.4f}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix (ma trận nhầm lẫn) ---")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Tải dữ liệu Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)

    print("5 dòng đặc trưng đầu tiên:")
    print(X[:5, :])
    print("\n5 nhãn đầu tiên:", y[:5])

    # 2. Chia tập train / test (giữ tỉ lệ lớp với stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )
    print(f"\nKích thước X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Mã hóa one-hot cho nhãn (phục vụ mô hình Keras)
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)

    # 3. Mô hình 1: MLPClassifier (scikit-learn)
    clf = MLPClassifier(
        solver="adam",
        alpha=1e-5,
        hidden_layer_sizes=(3, 3),
        activation="relu",
        max_iter=1000,
        random_state=1,
    )
    clf.fit(X_train, y_train)
    mlp_pred = clf.predict(X_test)

    print("\nNhãn dự báo (MLPClassifier):", mlp_pred)
    print("Nhãn thật:                  ", y_test)
    evaluate_model(y_test, mlp_pred, target_names, "MLPClassifier (Iris)")

    # 4. Mô hình 2: Mạng nơ-ron truyền thẳng với Keras
    network = models.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    print("\n--- Kiến trúc mạng Keras ---")
    network.summary()

    network.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n--- Huấn luyện mạng Keras ---")
    network.fit(X_train, train_labels, epochs=20, batch_size=40, verbose=1)

    test_loss, test_acc = network.evaluate(X_test, test_labels, verbose=0)
    print(f"\nKeras Test Loss: {test_loss:.4f} | Keras Test Accuracy: {test_acc:.4f}")

    # Chuyển xác suất dự đoán của Keras về chỉ số lớp (0, 1, 2) để đánh giá chi tiết
    keras_probs = network.predict(X_test, verbose=0)
    keras_pred = np.argmax(keras_probs, axis=1)
    evaluate_model(y_test, keras_pred, target_names, "Keras Sequential NN (Iris)")


if __name__ == "__main__":
    main()

# Bạn có muốn mình bổ sung thêm bước lưu mô hình (`.pkl` hoặc `.keras`) để dùng lại cho lần chạy sau trên PyCharm không?
# ### Lưu ý nhỏ khi chạy trên PyCharm
# * **Nếu `MLPClassifier` báo `ConvergenceWarning`:** Với tập dữ liệu nhỏ và `hidden_layer_sizes=(3, 3)`, tăng `max_iter` (ví dụ `max_iter=1000`) hoặc chuẩn hóa dữ liệu bằng `StandardScaler` sẽ giúp mô hình hội tụ ổn định hơn[cite: 27].
# * **Nếu file `tennis.csv` dùng tên cột khác:** Kiểm tra lại `data.columns` sau khi chạy để đảm bảo các cột `outlook`, `temp`, `humidity`, `windy`, `play` khớp chính xác[cite: 34].
#
# Bạn có muốn mình thêm phần đánh giá chi tiết (`classification_report`, ma trận nhầm lẫn) cho hai bài này không?
