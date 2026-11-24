import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def load_data(csv_path: str = "tennis.csv") -> pd.DataFrame:
    """Đọc dữ liệu tennis.csv từ máy."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{csv_path}'. Hãy đặt tennis.csv cùng thư mục với file .py hoặc đổi đường dẫn."
        )
    return pd.read_csv(csv_path)


def print_evaluation(y_true, y_pred, labels, title: str):
    """In độ chính xác, classification_report và ma trận nhầm lẫn."""
    print(f"\n==================== {title} ====================")
    print(f"Accuracy: {metrics.accuracy_score(y_true, y_pred):.4f}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("--- Confusion Matrix (ma trận nhầm lẫn) ---")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Greens")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.show()


def main():
    # Đổi thành đường dẫn tuyệt đối nếu cần, ví dụ: r"D:\Datasets\tennis.csv"
    data = load_data("tennis.csv")

    print("--- Thông tin dữ liệu ---")
    data.info()
    print("\nDanh sách cột:", data.columns.tolist())
    print("\n14 dòng dữ liệu:")
    print(data.head(14))

    # Mã hóa one-hot cho các thuộc tính điều kiện
    X = pd.get_dummies(
        data[["outlook", "temp", "humidity", "windy"]], dtype=int
    )
    y = data["play"]
    labels = sorted(y.unique().tolist())

    print("\n--- Đặc trưng sau khi mã hóa ---")
    print(X.head())
    print("Thứ tự cột đặc trưng:", X.columns.tolist())

    # 1. Huấn luyện trên toàn bộ dữ liệu & dự đoán mẫu mới (như bài thực hành)
    model_full = GaussianNB()
    model_full.fit(X, y)

    sample = pd.DataFrame(
        [[0, 1, 0, 0, 0, 1, 0, 1, 0]],
        columns=X.columns,
    )
    predicted = model_full.predict(sample)
    print("\nDự đoán cho mẫu đầu vào:", predicted)

    # Đánh giá trên toàn bộ 14 mẫu
    full_pred = model_full.predict(X)
    print_evaluation(y, full_pred, labels, "GaussianNB (Full 14 samples)")

    # 2. Chia train/test để đánh giá khách quan hơn trên mẫu chưa thấy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model_split = GaussianNB()
    model_split.fit(X_train, y_train)
    test_pred = model_split.predict(X_test)
    print_evaluation(y_test, test_pred, labels, "GaussianNB (Train/Test Split)")


if __name__ == "__main__":
    main()