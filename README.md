# 🏀 NBA 球員薪資等級預測系統（機器學習專案）

## 📌 專案簡介
本專案透過機器學習方法，根據球員數據（得分、助攻、籃板、效率值、年齡）預測其薪資等級：

- 底薪（Minimum）
- 中產（Mid-level）
- 頂薪（Max Contract）

⚠️ 本專案使用「模擬資料」，用於學習 ML 流程。

---

## 🧠 專案重點

本專案主要涵蓋以下技能：

- 資料前處理（StandardScaler）
- 訓練多種機器學習模型
- 模型評估（Accuracy / Precision / Recall）
- 混淆矩陣（Confusion Matrix）
- 簡單超參數分析（Validation Curve）
- 模型儲存與載入（pickle）

---

## 🤖 使用模型

- Baseline（DummyClassifier）
- KNN（K-Nearest Neighbors）
- Logistic Regression
- Random Forest
- SVM（Support Vector Machine）

---

## 📊 模型評估

使用以下指標評估模型：

- Accuracy
- Precision（macro avg）
- Recall（macro avg）

並輸出：

- 混淆矩陣（Heatmap）
- Validation Curve（Random Forest）

---


## 💡 怎麼升級和改進

方向1:
觀察：
- 模型容易將球員分類為中產（class 1）
- 可能原因為資料分佈集中於中間區間
- Random Forest 出現 overfitting 現象（training ≈1.0）

改進方向：
- 增加特徵（球隊戰績、上場時間等）
- 調整模型複雜度
- 使用 GridSearchCV

方向2:
改 RandomForest：

RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

用於防止 overfitting


## 🚀 執行方式

1️⃣ 安裝套件
pip install -r requirements.txt
2️⃣ 訓練模型
python train.py
3️⃣ 預測球員
python predict.py
🔮 預測範例
輸入：

[28.5, 7.2, 5.0, 26.5, 27]
輸出：

頂薪 (Max Contract)

⚠️ 專案限制
使用模擬資料（非真實 NBA 數據）

薪資標籤由公式生成（非真實市場機制）



