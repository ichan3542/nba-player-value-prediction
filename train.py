import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from data_utils import generate_nba_data, save_artifact


os.makedirs('outputs', exist_ok=True)


# 1. 準備數據
df = generate_nba_data()

X = df.drop('salary_level', axis=1)
y = df['salary_level']

# 標準化（對 KNN / SVM 很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 儲存 scaler
save_artifact(scaler, 'models/scaler.pkl')


# 2. 定義模型（含 baseline）
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

results = {}


# 3. 訓練 + 評估 + 混淆矩陣
plt.figure(figsize=(20, 4))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 儲存模型
    save_artifact(model, f'models/{name}.pkl')

    # 評估指標
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": report['macro avg']['precision'],
        "Recall": report['macro avg']['recall']
    }

    # 畫混淆矩陣
    plt.subplot(1, len(models), i + 1)
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Predicted 0', 'Predicted 1', 'Predicted 2'],
        yticklabels=['Actual 0', 'Actual 1', 'Actual 2']
    )
    plt.title(name)

plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
plt.show()


# 4. 顯示模型比較
print("Model Comparison:")
print(pd.DataFrame(results).T)


# 5. 簡單超參數分析（Random Forest）
param_range = [10, 50, 100]

train_scores, test_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=3
)

plt.figure()

plt.plot(param_range, np.mean(train_scores, axis=1), label="Training Score")
plt.plot(param_range, np.mean(test_scores, axis=1), label="Validation Score")

plt.title("Validation Curve (Random Forest)")
plt.xlabel("n_estimators")
plt.ylabel("Score")
plt.legend()

plt.savefig('outputs/validation_curve.png')
plt.show()
