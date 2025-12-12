#!/usr/bin/env python3
"""
🔧 距離正規化関数分析 実行スクリプト

カメラ距離による肩幅変化を分析し、正規化関数を生成するツール

使用例:
    # データ確認
    python run_normalization_analysis.py check "outputs/baseline/11月12日 1/4point_metrics.csv"
    
    # 1フレーム分析
    python run_normalization_analysis.py analyze_one "outputs/baseline/11月12日 1/4point_metrics.csv" "11月12日 1.mp4_frame0.jpg"
    
    # 全フレーム分析
    python run_normalization_analysis.py analyze_all "outputs/baseline/11月12日 1/4point_metrics.csv"
    
    # サンプル分析
    python run_normalization_analysis.py
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import traceback
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
import json

# --- 日本語フォント設定（Mac用） ---
plt.rcParams['font.family'] = 'AppleGothic'
# Windowsの場合は 'MS Gothic' や 'Meiryo' などに変更してください

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

try:
    from normalization_analysis.distance_normalization import DistanceNormalizationAnalyzer, check_available_data
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("📁 現在のディレクトリを確認してください")
    sys.exit(1)

def print_header():
    """🎨 ヘッダー表示"""
    print("🔧" + "=" * 60 + "🔧")
    print("📊      距離正規化関数分析ツール v1.0")
    print("🎯      カメラ距離による肩幅変化の定量化")
    print("🔧" + "=" * 60 + "🔧")

def get_file_emoji(file_type: str) -> str:
    """📁 ファイルタイプ別絵文字"""
    emoji_map = {
        'visualization': '📊',
        'report': '📝', 
        'function_data': '📋',
        'normalization_code': '🔧'
    }
    return emoji_map.get(file_type, '📄')

def extract_column_assignments_from_csv(csv_path):
    """
    CSVのcolumn_position列から自動で {列番号: [person_id, ...]} を抽出
    -1やNoneは除外
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    col_dict = {}
    for _, row in df.iterrows():
        try:
            col = int(float(row['column_position']))
            pid = int(row['person_id'])
            if col > 0:
                col_dict.setdefault(col, []).append(pid)
        except Exception:
            continue
    # 重複除去
    for k in col_dict:
        col_dict[k] = sorted(list(set(col_dict[k])))
    return col_dict

def create_analysis_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_shoulder_width_vs_column_with_fit(csv_path, output_dir):
    """
    距離-肩幅関係グラフを生成
    - 横軸: 列位置 (column_position)
    - 縦軸: 肩幅 (shoulder_width)
    - 個人データ: 点
    - 列平均: 赤い菱形
    - 最適指数減衰関数: 曲線（青）
    - 最適直線近似: 直線（緑点線）
    - パラメータは別々のjsonに保存
    - 正規化関数コードも別々のpythonファイルに保存
    """
    group_size = 1200
    interval = 40

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df[df['column_position'].apply(lambda x: pd.notnull(x) and float(x) > 0)]
    if 'column_position' not in df.columns or 'shoulder_width' not in df.columns or 'frame' not in df.columns:
        print("❌ 必要なカラムがありません")
        return

    df['frame_num'] = df['frame'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))))
                                        if isinstance(x, str) else int(x))
    df = df.sort_values('frame_num')
    max_frame = df['frame_num'].max()
    group_starts = list(range(0, max_frame + 1, group_size))
    columns = sorted(df['column_position'].unique())

    # 各列ごとにグループ中央値リスト
    col_group_medians = {col: [] for col in columns}

    # グループごとの中央値（灰色点）を集計
    for start in group_starts:
        end = start + group_size - 1
        group_df = df[(df['frame_num'] >= start) & (df['frame_num'] <= end)]
        required_count = group_size // interval
        if len(group_df) < required_count:
            continue
        for col in columns:
            col_df = group_df[group_df['column_position'] == col]
            if len(col_df) < required_count // len(columns):
                continue
            q1 = col_df['shoulder_width'].quantile(0.25)
            q3 = col_df['shoulder_width'].quantile(0.75)
            iqr_df = col_df[(col_df['shoulder_width'] >= q1) & (col_df['shoulder_width'] <= q3)]
            if iqr_df.empty:
                continue
            median = iqr_df['shoulder_width'].median()
            col_group_medians[col].append(median)

    # グラフ描画
    plt.figure(figsize=(10, 7))
    # グループごとの中央値（灰色点）
    for col in columns:
        for median in col_group_medians[col]:
            plt.scatter(col, median, color='gray', alpha=0.7, label='グループ中央値' if col == columns[0] and median == col_group_medians[columns[0]][0] else None)

    # 各列ごとのグループ中央値平均（赤菱・小さめ）
    xdata = []
    ydata = []
    for col in columns:
        medians = col_group_medians[col]
        if len(medians) > 0:
            xdata.append(col)
            ydata.append(np.mean(medians))
            plt.scatter(col, np.mean(medians), color='red', marker='D', s=40, label='列ごと中央値平均' if col == columns[0] else None)

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # --- 近似 ---
    # 指数減衰関数
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    fit_params_exp = None
    fit_params_lin = None
    try:
        popt_exp, pcov_exp = curve_fit(exp_decay, xdata, ydata, p0=(ydata.max(), 0.1, ydata.min()))
        fit_params_exp = popt_exp
        x_fit = np.linspace(xdata.min(), xdata.max(), 100)
        y_fit_exp = exp_decay(x_fit, *popt_exp)
        plt.plot(x_fit, y_fit_exp, color='blue', linewidth=2, label='指数減衰フィット（青）')
    except Exception as e:
        print(f"指数減衰フィット失敗: {e}")

    try:
        popt_lin = np.polyfit(xdata, ydata, 1)
        a_lin, b_lin = popt_lin
        fit_params_lin = (a_lin, b_lin, 0)
        y_fit_lin = a_lin * x_fit + b_lin
        plt.plot(x_fit, y_fit_lin, color='green', linestyle='dashed', linewidth=2, label='直線フィット（緑点線）')
    except Exception as e:
        print(f"直線フィット失敗: {e}")

    plt.xlabel('列位置 (column_position)', fontsize=13)
    plt.ylabel('肩幅 (px)', fontsize=13)
    plt.title('距離-肩幅関係（指数・直線フィット）', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)

    ymin = min([min(medians) for medians in col_group_medians.values() if medians]) if col_group_medians else 0
    ymax = max([max(medians) for medians in col_group_medians.values() if medians]) if col_group_medians else 100
    plt.ylim(ymin - 20, ymax + 40)
    plt.xticks([int(c) for c in columns])

    output_path = os.path.join(output_dir, "shoulder_width_vs_column_fit.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 距離-肩幅関係グラフを {output_path} に保存しました")

    info_path = os.path.join(output_dir, "analysis_info.txt")
    with open(info_path, "w", encoding="utf-8-sig") as f:
        f.write(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用CSV: {os.path.abspath(csv_path)}\n")
        f.write(f"列番号: {', '.join([str(c) for c in columns])}\n")
        f.write(f"グループ分割フレーム数: {group_size}\n")
        f.write(f"抽出間隔フレーム数: {interval}\n")
    print(f"✅ 分析情報を {info_path} に保存しました")

    # --- パラメータ保存（jsonを分ける） ---
    if fit_params_exp is not None:
        params_exp = {
            "a": fit_params_exp[0],
            "b": fit_params_exp[1],
            "c": fit_params_exp[2],
            "formula": "f(x) = a * exp(-b * x) + c"
        }
        with open(os.path.join(output_dir, "function_parameters_exp.json"), "w", encoding="utf-8-sig") as f:
            json.dump(params_exp, f, ensure_ascii=False, indent=2)
        print(f"✅ 指数減衰パラメータを function_parameters_exp.json に保存しました")

    if fit_params_lin is not None:
        params_lin = {
            "a": fit_params_lin[0],
            "b": fit_params_lin[1],
            "c": fit_params_lin[2],
            "formula": "f(x) = a * x + b + c"
        }
        with open(os.path.join(output_dir, "function_parameters_linear.json"), "w", encoding="utf-8-sig") as f:
            json.dump(params_lin, f, ensure_ascii=False, indent=2)
        print(f"✅ 直線パラメータを function_parameters_linear.json に保存しました")

    # --- 正規化関数コードも分けて保存 ---
    if fit_params_exp is not None:
        normalization_code_exp = f"""# 距離正規化関数（指数減衰）
import numpy as np

def distance_normalization_function_exp(column_position):
    \"\"\"
    指数減衰関数による肩幅予測
    f(x) = {fit_params_exp[0]:.6f} * exp(-{fit_params_exp[1]:.6f} * x) + {fit_params_exp[2]:.6f}
    \"\"\"
    return {fit_params_exp[0]:.6f} * np.exp(-{fit_params_exp[1]:.6f} * column_position) + {fit_params_exp[2]:.6f}

def normalize_shoulder_width_exp(measured_width, column_position, reference_column=1):
    \"\"\"
    実測肩幅を基準列で正規化（指数減衰）
    \"\"\"
    predicted_width = distance_normalization_function_exp(column_position)
    reference_width = distance_normalization_function_exp(reference_column)
    normalization_factor = reference_width / predicted_width
    return measured_width * normalization_factor
"""
        with open(os.path.join(output_dir, "normalization_function_exp.py"), "w", encoding="utf-8-sig") as f:
            f.write(normalization_code_exp)
        print(f"✅ 正規化関数コード（指数）を normalization_function_exp.py に保存しました")

    if fit_params_lin is not None:
        normalization_code_lin = f"""# 距離正規化関数（直線近似）
def distance_normalization_function_linear(column_position):
    \"\"\"
    直線近似による肩幅予測
    f(x) = {fit_params_lin[0]:.6f} * x + {fit_params_lin[1]:.6f} + {fit_params_lin[2]:.6f}
    \"\"\"
    return {fit_params_lin[0]:.6f} * column_position + {fit_params_lin[1]:.6f} + {fit_params_lin[2]:.6f}

def normalize_shoulder_width_linear(measured_width, column_position, reference_column=1):
    \"\"\"
    実測肩幅を基準列で正規化（直線近似）
    \"\"\"
    predicted_width = distance_normalization_function_linear(column_position)
    reference_width = distance_normalization_function_linear(reference_column)
    normalization_factor = reference_width / predicted_width
    return measured_width * normalization_factor
"""
        with open(os.path.join(output_dir, "normalization_function_linear.py"), "w", encoding="utf-8-sig") as f:
            f.write(normalization_code_lin)
        print(f"✅ 正規化関数コード（直線）を normalization_function_linear.py に保存しました")

# ...existing code...

def plot_shoulder_width_vs_column_with_fit_iqr(csv_path, output_dir):
    """
    距離-肩幅関係グラフ（IQR外れ値除去版）を生成
    - 横軸: 列位置 (column_position)
    - 縦軸: 肩幅 (shoulder_width)
    - 個人データ: 四分位範囲内のみ点表示
    - 列平均: 赤い菱形
    - 最適指数減衰関数: 曲線
    - 関数パラメータjson保存
    - 正規化関数コードも自動生成
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df[df['column_position'].apply(lambda x: pd.notnull(x) and float(x) > 0)]
    if 'column_position' not in df.columns or 'shoulder_width' not in df.columns:
        print("❌ 必要なカラムがありません")
        return

    # IQR外れ値除去
    iqr_mask = np.zeros(len(df), dtype=bool)
    for col in df['column_position'].unique():
        col_df = df[df['column_position'] == col]
        q1 = col_df['shoulder_width'].quantile(0.25)
        q3 = col_df['shoulder_width'].quantile(0.75)
        mask = (df['column_position'] == col) & (df['shoulder_width'] >= q1) & (df['shoulder_width'] <= q3)
        iqr_mask |= mask
    df_iqr = df[iqr_mask]

    plt.figure(figsize=(10, 7))
    plt.scatter(df_iqr['column_position'], df_iqr['shoulder_width'], alpha=0.5, label='IQR内個人データ')

    mean_df = df_iqr.groupby('column_position')['shoulder_width'].mean().reset_index()
    plt.scatter(mean_df['column_position'], mean_df['shoulder_width'], 
                color='red', marker='D', s=80, label='IQR内列平均')

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    xdata = mean_df['column_position']
    ydata = mean_df['shoulder_width']
    fit_params = None
    try:
        popt, pcov = curve_fit(exp_decay, xdata, ydata, p0=(ydata.max(), 0.1, ydata.min()))
        fit_params = popt
        x_fit = np.linspace(df['column_position'].min(), df['column_position'].max(), 100)
        y_fit = exp_decay(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='blue', linewidth=2, label='指数減衰フィット')
    except Exception as e:
        print(f"指数減衰フィット失敗: {e}")

    plt.xlabel('列位置 (column_position)', fontsize=13)
    plt.ylabel('肩幅 (px)', fontsize=13)
    plt.title('距離-肩幅関係（IQR外れ値除去）と正規化関数', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "shoulder_width_vs_column_fit_iqr.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 距離-肩幅関係グラフ（IQR外れ値除去）を {output_path} に保存しました")

def plot_angle_boxplot_by_column(csv_path, output_dir):
    """
    列位置ごとのなす角分布（箱ひげ図）を出力
    肩幅は個人ごとに固定し、両肩の中点→両耳の中点ベクトルと肩ベクトルのなす角を計算
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # -1やNoneは除外（float型にも対応）
    df = df[df['column_position'].apply(lambda x: pd.notnull(x) and float(x) > 0)]
    required_cols = [
        'column_position', 'person_id', 'shoulder_width',
        'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
        'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y'
    ]
    if not all(col in df.columns for col in required_cols):
        print("❌ 必要なカラムがありません")
        return

    # 肩幅を個人ごとに固定
    shoulder_width_dict = df.groupby('person_id')['shoulder_width'].mean().to_dict()

    def calc_angle_by_fixed_shoulder(row):
        pid = row['person_id']
        shoulder_width = shoulder_width_dict.get(pid, np.nan)
        lx, ly = row['left_shoulder_x'], row['left_shoulder_y']
        rx, ry = row['right_shoulder_x'], row['right_shoulder_y']
        center_x = (lx + rx) / 2
        center_y = (ly + ry) / 2
        shoulder_mid = np.array([center_x, center_y])
        shoulder_vec = np.array([shoulder_width, 0])  # 水平方向に肩幅分

        le_x, le_y = row['left_ear_x'], row['left_ear_y']
        re_x, re_y = row['right_ear_x'], row['right_ear_y']
        ear_mid = np.array([(le_x + re_x) / 2, (le_y + re_y) / 2])
        mid_vec = ear_mid - shoulder_mid

        dot = np.dot(shoulder_vec, mid_vec)
        norm_shoulder = np.linalg.norm(shoulder_vec)
        norm_mid = np.linalg.norm(mid_vec)
        if norm_shoulder == 0 or norm_mid == 0:
            return np.nan
        cos_theta = dot / (norm_shoulder * norm_mid)
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.degrees(np.arccos(cos_theta))

    df['shoulder_head_angle_fixed'] = df.apply(calc_angle_by_fixed_shoulder, axis=1)

    grouped = df.groupby('column_position')['shoulder_head_angle_fixed']
    data = []
    labels = []
    for col, group in grouped:
        values = group.dropna().values
        if len(values) > 0:
            data.append(values)
            labels.append(str(col))

    if len(data) == 0:
        print("⚠️ 箱ひげ図を描画するデータがありません")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='navy'),
                medianprops=dict(color='red'))
    plt.title('列位置ごとのなす角分布（肩幅固定・座標ベース・箱ひげ図）', fontsize=15)
    plt.xlabel('列位置 (column_position)', fontsize=13)
    plt.ylabel('なす角 (度)', fontsize=13)
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "angle_boxplot_by_column_fixed_vector.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 列位置ごとのなす角箱ひげ図（肩幅固定・座標ベース）を {output_path} に保存しました")

def check_command(args) -> int:
    """📊 データ確認コマンド"""
    print("🔍 利用可能データの確認中...")
    try:
        result = check_available_data(args.csv_path)
        if 'error' in result:
            print(f"❌ エラー: {result['error']}")
            return 1
        print(f"\n✅ データ確認結果:")
        print(f"📁 CSVファイル: {args.csv_path}")
        print(f"📊 総フレーム数: {result['total_frames']}")
        print(f"📏 肩幅データ列: {result['shoulder_width_column']}")
        print(f"\n📋 分析必要条件:")
        req = result['minimum_requirements']
        print(f"   最低必要人数: {req['min_people_per_analysis']}人")
        print(f"   最低必要列数: {req['min_columns']}列")
        print(f"   推奨人数/列: {req['recommended_people_per_column']}人")
        print(f"\n🎯 推奨フレーム（{req['min_people_per_analysis']}人以上検出）:")
        if not result['recommended_frames']:
            print("⚠️  条件を満たすフレームがありません")
            print("💡 より緩い条件のフレーム（上位5つ）:")
            all_frames = sorted(
                result['frame_details'].items(), 
                key=lambda x: x[1]['valid_shoulder_data'], 
                reverse=True
            )[:5]
            for frame, info in all_frames:
                print(f"   🎬 {frame}:")
                print(f"      検出: {info['valid_shoulder_data']}人")
                print(f"      ID: {info['available_ids']}")
                if info['shoulder_width_range']:
                    print(f"      肩幅範囲: {info['shoulder_width_range'][0]:.1f}-{info['shoulder_width_range'][1]:.1f}px")
                print()
        else:
            for frame in result['recommended_frames']:
                info = result['frame_details'][frame]
                print(f"   🎬 {frame}:")
                print(f"      検出: {info['valid_shoulder_data']}人")
                print(f"      利用可能ID: {info['available_ids']}")
                print(f"      肩幅範囲: {info['shoulder_width_range'][0]:.1f}-{info['shoulder_width_range'][1]:.1f}px")
                print()
        if result['recommended_frames']:
            frame = result['recommended_frames'][0]
            ids = result['frame_details'][frame]['available_ids']
            if len(ids) >= 6:
                ids_per_col = len(ids) // 3
                col1 = ids[:ids_per_col]
                col2 = ids[ids_per_col:ids_per_col*2] 
                col3 = ids[ids_per_col*2:]
                print(f"💡 推奨実行コマンド例:")
                print(f"python {Path(__file__).name} analyze_one \\")
                print(f'    "{args.csv_path}" \\')
                print(f'    "{frame}"')
        return 0
    except Exception as e:
        print(f"❌ 確認エラー: {e}")
        traceback.print_exc()
        return 1

def analyze_one_command(args) -> int:
    """🎯 1フレーム分析コマンド"""
    try:
        output_dir = create_analysis_output_dir(args.output_dir)
        column_assignments = extract_column_assignments_from_csv(args.csv_path)
        if not column_assignments:
            print("❌ エラー: column_position列が見つからないか、割り当てがありません")
            return 1
        if len(column_assignments) < 2:
            print("❌ エラー: 正規化関数作成には最低2列必要です")
            return 1
        print(f"🎯 1フレーム分析開始:")
        print(f"   📁 CSV: {args.csv_path}")
        print(f"   🎬 フレーム: {args.frame_id}")
        print(f"   📋 列構成: {column_assignments}")
        print(f"   📂 出力先: {output_dir}")
        analyzer = DistanceNormalizationAnalyzer(
            csv_path=args.csv_path,
            output_base_dir=output_dir
        )
        result = analyzer.analyze_distance_function(
            frame_id=args.frame_id,
            column_assignments=column_assignments
        )
        if result['success']:
            print(f"\n🎉 分析成功!")
            print(f"📁 結果フォルダ: {result['analysis_info']['output_dir']}")
            plot_shoulder_width_vs_column_with_fit(args.csv_path, output_dir)
            plot_shoulder_width_vs_column_with_fit_iqr(args.csv_path, output_dir)  # ←追加
            plot_angle_boxplot_by_column(args.csv_path, output_dir)
            return 0
        else:
            print(f"❌ 分析失敗: {result.get('error', '不明なエラー')}")
            return 1
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        traceback.print_exc()
        return 1

def analyze_all_command(args) -> int:
    """🎯 全フレーム分析コマンド"""
    try:
        output_dir = create_analysis_output_dir(args.output_dir)
        column_assignments = extract_column_assignments_from_csv(args.csv_path)
        if not column_assignments:
            print("❌ エラー: column_position列が見つからないか、割り当てがありません")
            return 1
        print(f"🎯 全フレーム分析開始:")
        print(f"   📁 CSV: {args.csv_path}")
        print(f"   📋 列構成: {column_assignments}")
        print(f"   📂 出力先: {output_dir}")
        plot_shoulder_width_vs_column_with_fit(args.csv_path, output_dir)
        plot_shoulder_width_vs_column_with_fit_iqr(args.csv_path, output_dir)  # ←追加
        plot_angle_boxplot_by_column(args.csv_path, output_dir)
        return 0
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        traceback.print_exc()
        return 1

def sample_command() -> int:
    """🔬 サンプル分析の実行"""
    print("🔬 サンプル正規化分析を実行...")
    try:
        sample_csv_paths = [
            'outputs/baseline/11月12日 1/4point_metrics.csv',
            'outputs/baseline/*/4point_metrics.csv',
            'data/4point_metrics.csv',
            '4point_metrics.csv'
        ]
        sample_csv = None
        for path_pattern in sample_csv_paths:
            if '*' in path_pattern:
                from glob import glob
                matches = glob(path_pattern)
                if matches:
                    sample_csv = matches[0]
                    break
            else:
                if Path(path_pattern).exists():
                    sample_csv = path_pattern
                    break
        if not sample_csv:
            print("❌ サンプル用CSVファイルが見つかりません")
            print("💡 以下のいずれかを配置してください:")
            for path in sample_csv_paths[:3]:
                print(f"   - {path}")
            return 1
        print(f"📁 サンプルファイル: {sample_csv}")
        output_dir = create_analysis_output_dir("outputs/normalization_analysis")
        plot_shoulder_width_vs_column_with_fit(sample_csv, output_dir)
        plot_angle_boxplot_by_column(sample_csv, output_dir)
        print(f"\n✅ サンプルグラフを {output_dir} に出力しました。")
        return 0
    except Exception as e:
        print(f"❌ サンプル分析エラー: {e}")
        traceback.print_exc()
        return 1

def main() -> int:
    """🎯 メイン関数"""
    print_header()
    parser = argparse.ArgumentParser(
        description='🔧 距離正規化関数分析ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # データ確認
  %(prog)s check "outputs/baseline/11月12日 1/4point_metrics.csv"
  
  # 1フレーム分析
  %(prog)s analyze_one "data.csv" "frame.jpg"
  
  # 全フレーム分析
  %(prog)s analyze_all "data.csv"
  
  # サンプル分析
  %(prog)s
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='実行コマンド')
    check_parser = subparsers.add_parser(
        'check', 
        help='📊 利用可能データの確認',
        description='CSVファイルから分析可能なフレーム・IDを確認'
    )
    check_parser.add_argument('csv_path', help='4点メトリクスCSVファイルのパス')
    analyze_one_parser = subparsers.add_parser(
        'analyze_one', 
        help='🎯 1フレーム分析',
        description='指定したフレームで正規化関数を生成'
    )
    analyze_one_parser.add_argument('csv_path', help='4点メトリクスCSVファイルのパス')
    analyze_one_parser.add_argument('frame_id', help='分析対象フレームID')
    analyze_one_parser.add_argument('--output-dir', default='outputs/normalization_analysis',
                               help='出力ディレクトリ (デフォルト: outputs/normalization_analysis)')
    analyze_all_parser = subparsers.add_parser(
        'analyze_all', 
        help='🎯 全フレーム分析',
        description='全フレームで分布・平均グラフを生成'
    )
    analyze_all_parser.add_argument('csv_path', help='4点メトリクスCSVファイルのパス')
    analyze_all_parser.add_argument('--output-dir', default='outputs/normalization_analysis',
                               help='出力ディレクトリ (デフォルト: outputs/normalization_analysis)')
    args = parser.parse_args()
    try:
        if args.command == 'check':
            return check_command(args)
        elif args.command == 'analyze_one':
            return analyze_one_command(args)
        elif args.command == 'analyze_all':
            return analyze_all_command(args)
        else:
            print("📝 引数なしでサンプル分析を実行します...")
            return sample_command()
    except KeyboardInterrupt:
        print("\n⏹️  ユーザーによる中断")
        return 130
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# --- 箱ひげ図の見方 ---
# 横軸：列位置（column_position）
# 縦軸：なす角（度）
# 箱：データの中央50%（第1四分位～第3四分位）
# 赤線：中央値
# ひげ：外れ値を除いた範囲
# 点：外れ値
# → 列ごとに姿勢（なす角）のばらつきや傾向が分かります