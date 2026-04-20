import pandas as pd
import numpy as np
import pickle
import os


def generate_nba_data():
    """
    生成模擬 NBA 球員數據（用於機器學習練習）
    """
    np.random.seed(42)
    n_players = 1000

    data = {
        'pts': np.random.uniform(5, 30, n_players),   # 得分
        'ast': np.random.uniform(1, 10, n_players),   # 助攻
        'reb': np.random.uniform(2, 12, n_players),   # 籃板
        'per': np.random.uniform(10, 30, n_players),  # 效率值
        'age': np.random.randint(19, 38, n_players)   # 年齡
    }

    df = pd.DataFrame(data)

    # 建立「球員綜合表現分數」+ 加入 noise 模擬真實世界
    score = (
        df['pts'] * 0.4 +
        df['per'] * 0.4 +
        df['ast'] * 0.1 +
        df['reb'] * 0.1 +
        np.random.normal(0, 2, n_players)  # 加入隨機噪音
    )

    # 分成三類（底薪 / 中產 / 頂薪）
    df['salary_level'] = pd.qcut(score, 3, labels=[0, 1, 2]).astype(int)

    return df


def save_artifact(obj, path):
    """
    儲存模型或 scaler
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_artifact(path):
    """
    載入模型或 scaler
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
