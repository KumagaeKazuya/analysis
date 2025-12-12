import numpy as np
import json
import os

def load_exponential_params(json_dir):
    """
    指定ディレクトリ内のfunction_parameters_exp.jsonから指数減衰関数パラメータ(a, b, c)を取得
    """
    json_path = os.path.join(json_dir, 'function_parameters_exp.json')
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        params = json.load(f)
    try:
        a = params['a']
        b = params['b']
        c = params['c']
        return a, b, c
    except Exception:
        raise ValueError("function_parameters_exp.jsonからパラメータ(a, b, c)を取得できません")

def load_linear_params(json_dir):
    """
    指定ディレクトリ内のfunction_parameters_linear.jsonから直線近似パラメータ(a, b, c)を取得
    """
    json_path = os.path.join(json_dir, 'function_parameters_linear.json')
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        params = json.load(f)
    try:
        a = params['a']
        b = params['b']
        c = params['c']
        return a, b, c
    except Exception:
        raise ValueError("function_parameters_linear.jsonからパラメータ(a, b, c)を取得できません")

def linear_func(distance, a, b, c):
    """
    距離補正用の直線関数
    distance: カメラからの距離や列番号
    a, b, c: fitting結果から取得したパラメータ
    """
    if distance is None:
        return None
    return a * distance + b * distance + c

def normalize_value_by_linear(value, distance, a, b, c, reference_distance=1):
    """
    value: 実測値（例：肩幅）
    distance: 測定位置（例：列番号）
    a, b, c: 直線関数パラメータ
    reference_distance: 基準位置（デフォルトは1列目）
    """
    predicted = linear_func(distance, a, b, c)
    reference = linear_func(reference_distance, a, b, c)
    if predicted is None or predicted == 0 or reference is None or reference == 0:
        return value
    normalization_factor = reference / predicted
    return value * normalization_factor

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