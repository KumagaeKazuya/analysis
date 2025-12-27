#!/usr/bin/env python3
"""
🔧 距離正規化関数分析 実行スクリプト（代表値CSV→階層回帰→関数フィット・可視化）

1. person_idの除外
2. 区間毎代表値CSVの保存・集約
3. id毎の代表値csvの正規性検定
4. 代表値CSVを用いた階層回帰による推定（列毎に保存されているID数やデータ数が異なるため）
5. 距離-肩幅関数フィット・パラメータjson出力
6. MSE計算
7. 分布の可視化（列毎の代表値と直線近似、指数減衰フィットを描画したグラフの作成）

使い方:
    python run_normalization2.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import json
import seaborn as sns
import shutil
import glob
from scipy.stats import shapiro
import statsmodels.formula.api as smf

plt.rcParams['font.family'] = 'AppleGothic'

def select_valid_ids_interactive():
    print("=== person_id選別用CSVファイルのパスを入力してください ===")
    id_csv_path = input("例: summary.csv > ").strip()
    id_df = pd.read_csv(id_csv_path)
    if not {'person_id', 'frames_with_id', 'total_frames'}.issubset(id_df.columns):
        print("❌ 必要なカラムがありません（person_id, frames_with_id, total_frames）")
        sys.exit(1)
    id_df['rate'] = id_df['frames_with_id'] / id_df['total_frames']
    valid_ids = id_df.loc[id_df['rate'] >= 0.5, 'person_id'].unique()
    print(f"50%以上のperson_id: {list(valid_ids)}")
    if len(valid_ids) == 0:
        print("⚠️ 条件を満たすperson_idがありません")
        sys.exit(1)
    print("=== 特例で除外したいperson_idがあればカンマ区切りで入力してください（例: 3,7,15）===")
    print("除外しない場合は何も入力せずEnterを押してください")
    exclude_input = input("除外person_id: ").strip()
    if exclude_input:
        try:
            exclude_ids = [int(x) for x in exclude_input.split(",") if x.strip().isdigit()]
            print(f"除外するperson_id: {exclude_ids}")
            valid_ids = [pid for pid in valid_ids if pid not in exclude_ids]
        except Exception as e:
            print(f"⚠️ 除外IDの解析に失敗しました: {e}")
    print(f"最終的に使用するperson_id: {list(valid_ids)}")
    if len(valid_ids) == 0:
        print("⚠️ 除外後、条件を満たすperson_idがありません")
        sys.exit(1)
    print("=== データ取得用CSVファイルのパスを入力してください ===")
    data_csv_path = input("例: all_data.csv > ").strip()
    data_df = pd.read_csv(data_csv_path)
    filtered_df = data_df[data_df['person_id'].isin(valid_ids)].copy()
    print(f"抽出後のデータ件数: {len(filtered_df)}")
    return filtered_df

def save_median_rows_per_minute(
    df, id_col='person_id', frame_col='frame', shoulder_col='shoulder_width',
    out_dir='median_rows_by_minute', frames_per_minute=1200
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for pid in df[id_col].unique():
        df_pid = df[df[id_col] == pid].copy()
        if df_pid.empty:
            continue
        df_pid[frame_col] = df_pid[frame_col].astype(str).str.extract(r'(\d+)')[0]
        frames = pd.to_numeric(df_pid[frame_col], errors='coerce').dropna().astype(int)
        df_pid = df_pid.loc[frames.index]
        df_pid[frame_col] = frames
        frame_max = frames.max()
        bins = np.arange(40, frame_max + frames_per_minute, frames_per_minute)
        n_bins = len(bins)
        median_rows = []
        for i in range(n_bins):
            start = bins[i]
            end = start + frames_per_minute - 1
            if start < 40:
                continue
            if end > frame_max:
                if (frame_max - start + 1) < frames_per_minute:
                    continue
                end = frame_max
            df_bin = df_pid[(df_pid[frame_col] >= start) & (df_pid[frame_col] <= end)]
            if df_bin.empty or (end - start + 1) < frames_per_minute:
                continue
            median_val = df_bin[shoulder_col].median()
            median_idx = (df_bin[shoulder_col] - median_val).abs().idxmin()
            median_row = df_bin.loc[median_idx]
            median_row = median_row.copy()
            median_row['minute_bin'] = f"{int(start)}-{int(end)}"
            median_rows.append(median_row)
        if median_rows:
            median_df = pd.DataFrame(median_rows)
            person_dir = os.path.join(out_dir, f"person_{pid}")
            os.makedirs(person_dir, exist_ok=True)
            median_df.to_csv(os.path.join(person_dir, f"median_rows_pid{pid}.csv"), index=False)
            print(f"person_id={pid} 区間ごとの中央値行を {person_dir}/median_rows_pid{pid}.csv に保存しました。")

def aggregate_by_column_position(median_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(median_dir, "person_*", "median_rows_pid*.csv"))
    if not all_files:
        print("代表値CSVが見つかりません")
        return
    df_all = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    if 'column_position' not in df_all.columns:
        print("column_positionカラムがありません")
        return
    for col in sorted(df_all['column_position'].unique()):
        df_col = df_all[df_all['column_position'] == col]
        out_path = os.path.join(out_dir, f"column_{col}_median_rows.csv")
        df_col.to_csv(out_path, index=False)
        print(f"列 {col} の代表値を {out_path} に保存しました")

def check_normality_by_column(column_dir, columns=None):
    if columns is None:
        files = glob.glob(os.path.join(column_dir, "column_*_median_rows.csv"))
        # 小数点付きの列名に対応
        columns = sorted([float(os.path.basename(f).split('_')[1]) for f in files])
    for col in columns:
        # ファイル名も小数点付きで探索
        col_str = str(col)
        csv_path = os.path.join(column_dir, f"column_{col_str}_median_rows.csv")
        if not os.path.exists(csv_path):
            print(f"列{col}のCSVが見つかりません: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty or 'shoulder_width' not in df.columns:
            print(f"列{col}のデータが不正です")
            continue
        plt.figure(figsize=(8,5))
        sns.histplot(df['shoulder_width'], kde=True, bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Column {col} Shoulder Width Distribution')
        plt.xlabel('Shoulder Width')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(column_dir, f"column_{col_str}_shoulder_width_hist_kde.png"))
        plt.close()
        print(f"ヒストグラム・KDEを {os.path.join(column_dir, f'column_{col_str}_shoulder_width_hist_kde.png')} に保存しました")
        stat, p = shapiro(df['shoulder_width'])
        print(f"列{col} Shapiro-Wilk検定 p値: {p:.4f}")
        if p > 0.05:
            print("→ 正規分布とみなせます")
        else:
            print("→ 正規分布ではありません")

def fit_mixed_effect_and_save_params_from_csv(column_dir, output_dir):
    """
    列ごと代表値CSVを使って階層回帰→関数フィット・パラメータjson出力
    """
    files = glob.glob(os.path.join(column_dir, "column_*_median_rows.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'column_position' not in df.columns or 'person_id' not in df.columns or 'shoulder_width' not in df.columns:
            continue
        dfs.append(df[['column_position', 'person_id', 'shoulder_width']])
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['column_position'] = df_all['column_position'].astype(int)
    # 階層回帰
    print("階層回帰（mixed effect model）を実行します...")
    model = smf.mixedlm("shoulder_width ~ C(column_position)", df_all, groups=df_all["person_id"])
    result = model.fit()
    print(result.summary())
    columns = sorted(df_all['column_position'].unique())
    pred_df = pd.DataFrame({'column_position': columns})
    intercept = result.params['Intercept']
    pred_df['shoulder_width_pred'] = intercept
    for col in columns:
        if col == columns[0]:
            continue
        key = f'C(column_position)[T.{col}]'
        if key in result.params:
            pred_df.loc[pred_df['column_position'] == col, 'shoulder_width_pred'] += result.params[key]
    print("階層回帰による列ごとの推定値:")
    print(pred_df)
    # 関数フィット
    xdata = pred_df['column_position'].values
    ydata = pred_df['shoulder_width_pred'].values
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    fit_params_exp = None
    fit_params_lin = None
    try:
        popt_exp, _ = curve_fit(exp_decay, xdata, ydata, p0=(ydata.max(), 0.1, ydata.min()))
        fit_params_exp = popt_exp
    except Exception as e:
        print(f"指数減衰フィット失敗: {e}")
    try:
        popt_lin = np.polyfit(xdata, ydata, 1)
        a_lin, b_lin = popt_lin
        fit_params_lin = (a_lin, b_lin, 0)
    except Exception as e:
        print(f"直線フィット失敗: {e}")
    def save_params(params, name, formula):
        if params is not None:
            d = {
                "a": params[0],
                "b": params[1],
                "c": params[2],
                "formula": formula
            }
            with open(os.path.join(output_dir, f"function_parameters_{name}.json"), "w", encoding="utf-8-sig") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            print(f"✅ {name}パラメータを function_parameters_{name}.json に保存しました")
    save_params(fit_params_exp, "exp_mixed", "f(x) = a * exp(-b * x) + c")
    save_params(fit_params_lin, "linear_mixed", "f(x) = a * x + b + c")
    # グラフ描画
    plt.figure(figsize=(10, 7))
    plt.scatter(pred_df['column_position'], pred_df['shoulder_width_pred'], color='red', marker='D', s=80, label='階層回帰推定値')
    x_fit = np.linspace(xdata.min(), xdata.max(), 100)
    if fit_params_exp is not None:
        plt.plot(x_fit, exp_decay(x_fit, *fit_params_exp), color='blue', linewidth=2, label='指数減衰フィット')
    if fit_params_lin is not None:
        plt.plot(x_fit, fit_params_lin[0]*x_fit + fit_params_lin[1] + fit_params_lin[2], color='green', linestyle='dashed', linewidth=2, label='直線フィット')
    plt.xlabel('列位置 (column_position)', fontsize=13)
    plt.ylabel('肩幅 (px)', fontsize=13)
    plt.title('距離-肩幅関係（階層回帰推定値・近似）', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"shoulder_width_vs_column_fit_mixed.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 距離-肩幅関係グラフ（階層回帰推定値）を {output_path} に保存しました")
    return fit_params_exp, fit_params_lin, pred_df

def calc_and_save_mse(pred_df, fit_params_exp, fit_params_lin, output_dir, suffix=""):
    if fit_params_exp is None or fit_params_lin is None:
        print("⚠️ 近似パラメータが不足しているためMSE計算をスキップします。")
        return
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    def linear_func(x, a, b, c):
        return a * x + b + c
    x = pred_df['column_position'].values
    y = pred_df['shoulder_width_pred'].values
    y_pred_exp = exp_func(x, *fit_params_exp)
    y_pred_lin = linear_func(x, *fit_params_lin)
    mse_exp = np.mean((y - y_pred_exp) ** 2)
    mse_lin = np.mean((y - y_pred_lin) ** 2)
    result_text = (
        f"直線近似のMSE{suffix}: {mse_lin:.3f}\n"
        f"指数近似のMSE{suffix}: {mse_exp:.3f}\n"
    )
    print(result_text)
    with open(os.path.join(output_dir, f"mse_result{suffix}.txt"), "w", encoding="utf-8-sig") as f:
        f.write(result_text)

def main():
    print("🔧 距離正規化関数分析ツール（代表値CSV→階層回帰→関数フィット・可視化）")
    filtered_df = select_valid_ids_interactive()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"analysis_{timestamp}"

    # 2. 区間毎代表値CSVの保存
    median_dir = os.path.join(output_dir, 'median_rows_by_minute')
    save_median_rows_per_minute(
        filtered_df,
        id_col='person_id',
        frame_col='frame',
        shoulder_col='shoulder_width',
        out_dir=median_dir,
        frames_per_minute=1200
    )

    # 2. 列ごとに代表値を集約して保存
    column_dir = os.path.join(output_dir, 'column_median_rows')
    aggregate_by_column_position(
        median_dir=median_dir,
        out_dir=column_dir
    )

    # 3. id毎の代表値csvの正規性検定
    check_normality_by_column(column_dir)

    # 4. 代表値CSVを用いた階層回帰による推定
    fit_params_exp, fit_params_lin, pred_df = fit_mixed_effect_and_save_params_from_csv(column_dir, output_dir)

    # 5. 距離-肩幅関数フィット・パラメータjson出力（上記で実施済み）

    # 6. MSE計算
    if pred_df is not None:
        calc_and_save_mse(pred_df, fit_params_exp, fit_params_lin, output_dir, suffix="_mixed")

if __name__ == "__main__":
    main()