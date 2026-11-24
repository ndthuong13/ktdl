import os
import math
import pandas as pd


def load_data(csv_path: str = "PlayTennis.csv") -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV trên máy."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{csv_path}'. Hãy đặt PlayTennis.csv vào cùng thư mục với file .py hoặc sửa lại đường dẫn."
        )
    return pd.read_csv(csv_path)


def find_entropy(series: pd.Series) -> float:
    """Tính Entropy của một Series (biến mục tiêu hoặc thuộc tính)."""
    total = len(series)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in series.value_counts():
        p = count / total
        if p > 0:
            entropy += -p * math.log2(p)
    return round(entropy, 3)


def information_gain(data: pd.DataFrame, feature_series: pd.Series, target_col: str = "play") -> float:
    """Tính Information Gain khi chia dữ liệu theo một thuộc tính."""
    total_rows = len(data)
    if total_rows == 0:
        return 0.0

    weighted_child_entropy = 0.0
    for val in feature_series.unique():
        subset = data[feature_series == val]
        weight = len(subset) / total_rows
        child_entropy = find_entropy(subset[target_col])
        weighted_child_entropy += weight * child_entropy

    parent_entropy = find_entropy(data[target_col])
    return round(parent_entropy - weighted_child_entropy, 3)


def analyze_feature(data: pd.DataFrame, feature: str, target_col: str = "play") -> float:
    """In thông tin phân nhóm, entropy từng giá trị và information gain của thuộc tính."""
    print(f"\n--- Thuộc tính: {feature} ---")
    for val in data[feature].unique():
        subset = data[data[feature] == val]
        if subset.empty:
            continue
        class_counts = subset[target_col].value_counts().to_dict()
        val_entropy = find_entropy(subset[target_col])
        print(f"  + {feature} = {val} (số mẫu: {len(subset)}, phân bố: {class_counts}) -> Entropy = {val_entropy}")

    ig = information_gain(data, data[feature], target_col=target_col)
    print(f"=> Information Gain cho '{feature}' = {ig}")
    return ig


def main():
    # Có thể đổi thành đường dẫn tuyệt đối, ví dụ: r"D:\Datasets\PlayTennis.csv"
    data = load_data("PlayTennis.csv")

    print("Kích thước dữ liệu:", data.shape)
    print("\n5 dòng đầu:")
    print(data.head())

    print("\nSố lượng giá trị duy nhất từng cột:")
    print(data.nunique())

    print("\nPhân bố biến mục tiêu 'play':")
    print(data["play"].value_counts())
    print(f"Entropy toàn bộ tập dữ liệu: {find_entropy(data['play'])}")

    # Bước 1: Tính trên toàn bộ tập dữ liệu
    print("\n=== BƯỚC 1: TÍNH TRÊN TOÀN BỘ DỮ LIỆU ===")
    for col in ["outlook", "temp", "humidity", "windy"]:
        analyze_feature(data, col)

    # Bước 2 & 3: Nhánh outlook == 'sunny'
    print("\n=== NHÁNH OUTLOOK = SUNNY ===")
    sunny = data[data["outlook"] == "sunny"]
    print(f"Entropy tập Sunny: {find_entropy(sunny['play'])}")
    for col in ["temp", "humidity", "windy"]:
        analyze_feature(sunny, col)

    # Nhánh outlook == 'rainy'
    print("\n=== NHÁNH OUTLOOK = RAINY ===")
    rainy = data[data["outlook"] == "rainy"]
    print(f"Entropy tập Rainy: {find_entropy(rainy['play'])}")
    for col in ["temp", "humidity", "windy"]:
        analyze_feature(rainy, col)


if __name__ == "__main__":
    main()