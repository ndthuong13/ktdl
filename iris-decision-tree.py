#pip install pandas numpy scikit-learn seaborn matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    # 1. Nạp dữ liệu Iris
    iris = datasets.load_iris()
    feature_cols = ["Petal length", "Petal Width", "Sepal Length", "Sepal Width"]
    data = pd.DataFrame(iris["data"], columns=feature_cols)

    data["Species_name"] = [iris["target_names"][i] for i in iris["target"]]
    print("5 dòng đầu:")
    print(data.head())
    print("\nCác loài hoa:", data["Species_name"].unique())

    # 2. Khám phá dữ liệu (EDA)
    sns.pairplot(data, hue="Species_name")
    plt.tight_layout()
    plt.show()

    # Chuyển biến mục tiêu: versicolor -> 1, các loài còn lại -> 0
    data["Species"] = data["Species_name"].map({
        "versicolor": 1,
        "setosa": 0,
        "virginica": 0,
    })
    print("\nPhân bố nhãn mục tiêu (1: versicolor, 0: khác):")
    print(data["Species"].value_counts())

    # 3. Chia tập Train / Test
    X = data[["Sepal Length", "Sepal Width", "Petal length", "Petal Width"]].values
    y = data["Species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=1
    )

    # 4. Huấn luyện mô hình cây quyết định
    clf = DecisionTreeClassifier(random_state=12)
    clf.fit(X_train, y_train)

    # 5. Dự đoán và đánh giá
    prediction = clf.predict(X_test)
    print("\nKết quả dự đoán trên tập test:")
    print(prediction)

    print(f"\nAccuracy: {metrics.accuracy_score(y_test, prediction):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, prediction))

    # 6. Vẽ Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.title("Confusion Matrix - Iris (Versicolor vs Others)")
    plt.tight_layout()
    plt.show()

    # (Tùy chọn) Vẽ sơ đồ cây quyết định
    plt.figure(figsize=(12, 6))
    plot_tree(
        clf,
        feature_names=["Sepal Length", "Sepal Width", "Petal length", "Petal Width"],
        class_names=["Other (0)", "Versicolor (1)"],
        filled=True,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()