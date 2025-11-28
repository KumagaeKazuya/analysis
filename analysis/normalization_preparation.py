import numpy as np
import json
import os

def load_exponential_params(json_dir):
    """
    指定ディレクトリ内のfunction_parameters.jsonから指数減衰関数パラメータ(a, b, c)を取得
    """
    json_path = os.path.join(json_dir, 'function_parameters.json')
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        params = json.load(f)
    # 修正: 階層を直接取得
    try:
        a = params['parameters']['a']
        b = params['parameters']['b']
        c = params['parameters']['c']
        return a, b, c
    except Exception:
        raise ValueError("function_parameters.jsonからパラメータ(a, b, c)を取得できません")

def exponential_decay(distance, a, b, c):
    """
    距離補正用の指数減衰関数
    distance: カメラからの距離や列番号
    a, b, c: fitting結果から取得したパラメータ
    """
    if distance is None or distance == 0:
        return None
    return a * np.exp(-b * distance) + c

def normalize_value_by_decay(value, distance, a, b, c, reference_distance=1):
    """
    value: 実測値（例：肩幅）
    distance: 測定位置（例：列番号）
    a, b, c: 減少関数パラメータ
    reference_distance: 基準位置（デフォルトは1列目）
    """
    predicted = exponential_decay(distance, a, b, c)
    reference = exponential_decay(reference_distance, a, b, c)
    if predicted is None or predicted == 0 or reference is None or reference == 0:
        return value
    normalization_factor = reference / predicted
    return value * normalization_factor