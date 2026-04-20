import sys
import numpy as np
import pandas as pd
from data_utils import load_artifact


def _ensure_utf8_output():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def predict_player_salary(model_name, stats):
    """
    預測球員薪資等級

    stats = [pts, ast, reb, per, age]
    """
    try:
        model = load_artifact(f'models/{model_name}.pkl')
        scaler = load_artifact('models/scaler.pkl')

        feature_columns = ["pts", "ast", "reb", "per", "age"]
        stats_df = pd.DataFrame([stats], columns=feature_columns)
        scaled_stats = scaler.transform(stats_df)
        pred = model.predict(scaled_stats)[0]

        mapping = {
            0: "底薪 (Minimum)",
            1: "中產 (Mid-level)",
            2: "頂薪 (Max Contract)"
        }

        return mapping[pred]

    except FileNotFoundError:
        return "❌ 找不到模型，請先執行 train.py"


# 測試資料（模擬球員）
if __name__ == "__main__":
    _ensure_utf8_output()
    new_player = [26.5, 7.2, 5.0, 16.5, 27]
    selected_model = "RandomForest"

    result = predict_player_salary(selected_model, new_player)
    print(f"使用 {selected_model} 預測結果: {result}")
