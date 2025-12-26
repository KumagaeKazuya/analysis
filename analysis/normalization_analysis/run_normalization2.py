#!/usr/bin/env python3
"""
🔧 距離正規化関数分析 実行スクリプト（person_id選別・除外ID指定・2種グラフ・MSE計算・KDE/ヒストグラム分布可視化）

1. summary.csv（person_id, frames_with_id, total_frames）でperson_idを選別（30%以上のみ）
2. 除外したいperson_idを対話的に指定可能
3. データCSVから該当person_idのみ抽出
4. 距離-肩幅グラフ（全データ/IQR除去）・2種近似・MSE計算・KDE/ヒストグラム分布可視化（全データ/IQR除去）を実行

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

plt.rcParams['font.family'] = 'AppleGothic'

def select_valid_ids_interactive():
    print("=== person_id選別用CSVファイルのパスを入力してください ===")
    id_csv_path = input("例: summary.csv > ").strip()
    id_df = pd.read_csv(id_csv_path)
    if not {'person_id', 'frames_with_id', 'total_frames'}.issubset(id_df.columns):
        print("❌ 必要なカラムがありません（person_id, frames_with_id, total_frames）")
        sys.exit(1)
    id_df['rate'] = id_df['frames_with_id'] / id_df['total_frames']
    valid_ids = id_df.loc[id_df['rate'] >= 0.3, 'person_id'].unique()
    print(f"30%以上のperson_id: {list(valid_ids)}")
    if len(valid_ids) == 0:
        print("⚠️ 条件を満たすperson_idがありません")
        sys.exit(1)

    # --- 除外IDの指定 ---
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

def plot_shoulder_width_vs_column_with_fit(df, output_dir):
    """
    距離-肩幅関係グラフを2種類（全データ/IQR除去）生成し、2種近似・パラメータ保存
    ※全データは中央値、IQR除去後は平均値で代表値を取得
    """
    if 'column_position' not in df.columns or 'shoulder_width' not in df.columns:
        print("❌ 必要なカラムがありません")
        return None, None, None, None, None, None

    os.makedirs(output_dir, exist_ok=True)

    def plot_and_fit_median(plot_df, suffix, title_add):
        median_df = plot_df.groupby('column_position')['shoulder_width'].median().reset_index()
        xdata = median_df['column_position'].values
        ydata = median_df['shoulder_width'].values

        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        fit_params_exp = None
        fit_params_lin = None
        x_fit = np.linspace(xdata.min(), xdata.max(), 100)
        try:
            popt_exp, _ = curve_fit(exp_decay, xdata, ydata, p0=(ydata.max(), 0.1, ydata.min()))
            fit_params_exp = popt_exp
            y_fit_exp = exp_decay(x_fit, *popt_exp)
        except Exception as e:
            print(f"指数減衰フィット失敗: {e}")
            y_fit_exp = None

        try:
            popt_lin = np.polyfit(xdata, ydata, 1)
            a_lin, b_lin = popt_lin
            fit_params_lin = (a_lin, b_lin, 0)
            y_fit_lin = a_lin * x_fit + b_lin
        except Exception as e:
            print(f"直線フィット失敗: {e}")
            y_fit_lin = None

        # 近似グラフ描画
        plt.figure(figsize=(10, 7))
        plt.scatter(plot_df['column_position'], plot_df['shoulder_width'], color='gray', alpha=0.5, label='個人データ')
        plt.scatter(median_df['column_position'], median_df['shoulder_width'], color='red', marker='D', s=60, label='列ごと中央値')
        if y_fit_exp is not None:
            plt.plot(x_fit, y_fit_exp, color='blue', linewidth=2, label='指数減衰フィット（青）')
        if y_fit_lin is not None:
            plt.plot(x_fit, y_fit_lin, color='green', linestyle='dashed', linewidth=2, label='直線フィット（緑点線）')
        plt.xlabel('列位置 (column_position)', fontsize=13)
        plt.ylabel('肩幅 (px)', fontsize=13)
        plt.title('距離-肩幅関係（中央値・近似）', fontsize=15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"shoulder_width_vs_column_fit{suffix}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ 距離-肩幅関係グラフを {output_path} に保存しました")

        return fit_params_exp, fit_params_lin

    def plot_and_fit_mean(plot_df, suffix, title_add):
        mean_df = plot_df.groupby('column_position')['shoulder_width'].mean().reset_index()
        xdata = mean_df['column_position'].values
        ydata = mean_df['shoulder_width'].values

        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        fit_params_exp = None
        fit_params_lin = None
        x_fit = np.linspace(xdata.min(), xdata.max(), 100)
        try:
            popt_exp, _ = curve_fit(exp_decay, xdata, ydata, p0=(ydata.max(), 0.1, ydata.min()))
            fit_params_exp = popt_exp
            y_fit_exp = exp_decay(x_fit, *popt_exp)
        except Exception as e:
            print(f"指数減衰フィット失敗: {e}")
            y_fit_exp = None

        try:
            popt_lin = np.polyfit(xdata, ydata, 1)
            a_lin, b_lin = popt_lin
            fit_params_lin = (a_lin, b_lin, 0)
            y_fit_lin = a_lin * x_fit + b_lin
        except Exception as e:
            print(f"直線フィット失敗: {e}")
            y_fit_lin = None

        # 近似グラフ描画
        plt.figure(figsize=(10, 7))
        plt.scatter(plot_df['column_position'], plot_df['shoulder_width'], color='gray', alpha=0.5, label='個人データ')
        plt.scatter(mean_df['column_position'], mean_df['shoulder_width'], color='orange', marker='D', s=60, label='列ごと平均値')
        if y_fit_exp is not None:
            plt.plot(x_fit, y_fit_exp, color='blue', linewidth=2, label='指数減衰フィット（青）')
        if y_fit_lin is not None:
            plt.plot(x_fit, y_fit_lin, color='green', linestyle='dashed', linewidth=2, label='直線フィット（緑点線）')
        plt.xlabel('列位置 (column_position)', fontsize=13)
        plt.ylabel('肩幅 (px)', fontsize=13)
        plt.title('距離-肩幅関係（IQR除去・平均値・近似）', fontsize=15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"shoulder_width_vs_column_fit{suffix}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ 距離-肩幅関係グラフ（IQR除去）を {output_path} に保存しました")

        return fit_params_exp, fit_params_lin

    # 全データ（中央値）
    fit_params_exp_all, fit_params_lin_all = plot_and_fit_median(df, "", "")

    # IQR除去グラフ（平均値で代表値）
    iqr_mask = np.zeros(len(df), dtype=bool)
    for col in df['column_position'].unique():
        col_df = df[df['column_position'] == col]
        q1 = col_df['shoulder_width'].quantile(0.25)
        q3 = col_df['shoulder_width'].quantile(0.75)
        mask = (df['column_position'] == col) & (df['shoulder_width'] >= q1) & (df['shoulder_width'] <= q3)
        iqr_mask |= mask
    df_iqr = df[iqr_mask]

    fit_params_exp_iqr, fit_params_lin_iqr = plot_and_fit_mean(df_iqr, "_iqr", "（IQR除去・平均値）")

    # パラメータ保存（全データ/IQR両方）
    def save_params(params, name):
        if params is not None:
            d = {
                "a": params[0],
                "b": params[1],
                "c": params[2],
                "formula": "f(x) = a * exp(-b * x) + c" if "exp" in name else "f(x) = a * x + b + c"
            }
            with open(os.path.join(output_dir, f"function_parameters_{name}.json"), "w", encoding="utf-8-sig") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            print(f"✅ {name}パラメータを function_parameters_{name}.json に保存しました")

    save_params(fit_params_exp_all, "exp_all")
    save_params(fit_params_lin_all, "linear_all")
    save_params(fit_params_exp_iqr, "exp_iqr")
    save_params(fit_params_lin_iqr, "linear_iqr")

    return fit_params_exp_all, fit_params_lin_all, fit_params_exp_iqr, fit_params_lin_iqr, df, df_iqr

def calc_and_save_mse(df, fit_params_exp, fit_params_lin, output_dir, suffix=""):
    """
    直線近似・指数近似のMSEを計算し、output_dirに保存
    """
    if fit_params_exp is None or fit_params_lin is None:
        print("⚠️ 近似パラメータが不足しているためMSE計算をスキップします。")
        return

    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    def linear_func(x, a, b, c):
        return a * x + b + c

    x = df['column_position'].values
    y = df['shoulder_width'].values
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

def plot_kde_by_column(df, output_dir, suffix=""):
    """
    各列(column_position)ごとのshoulder_width分布をKDEのみで可視化
    """
    if 'column_position' not in df.columns or 'shoulder_width' not in df.columns:
        print("❌ 必要なカラムがありません")
        return

    plt.figure(figsize=(12, 7))
    columns = sorted(df['column_position'].unique())
    colors = sns.color_palette("husl", len(columns))
    for i, col in enumerate(columns):
        data = df[df['column_position'] == col]['shoulder_width'].values
        if len(data) > 1:
            sns.kdeplot(data, label=f'列 {col}', color=colors[i], fill=True, alpha=0.3)
        else:
            plt.scatter(data, [0], label=f'列 {col}', color=colors[i])

    # 5ごとの目盛と補助線
    x_min = int(df['shoulder_width'].min()) // 5 * 5
    x_max = int(df['shoulder_width'].max()) // 5 * 5 + 5
    plt.xticks(np.arange(x_min, x_max + 1, 5))
    plt.grid(True, alpha=0.3)
    for x in np.arange(x_min, x_max + 1, 5):
        plt.axvline(x, color='gray', linestyle='dotted', linewidth=0.7, alpha=0.5)

    plt.xlabel('肩幅 (px)', fontsize=13)
    plt.ylabel('密度', fontsize=13)
    plt.title(f'各列ごとの肩幅分布（KDE）{"（IQR除去）" if suffix == "_iqr" else ""}', fontsize=15)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"shoulder_width_kde{suffix}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 各列ごとの肩幅分布（KDE）を {output_path} に保存しました")

def plot_hist_by_column(df, output_dir, suffix=""):
    """
    各列(column_position)ごとのshoulder_width分布をヒストグラムのみで可視化
    """
    if 'column_position' not in df.columns or 'shoulder_width' not in df.columns:
        print("❌ 必要なカラムがありません")
        return

    plt.figure(figsize=(12, 7))
    columns = sorted(df['column_position'].unique())
    colors = sns.color_palette("husl", len(columns))
    bins = np.arange(int(df['shoulder_width'].min()) // 5 * 5,
                     int(df['shoulder_width'].max()) // 5 * 5 + 10, 5)
    for i, col in enumerate(columns):
        data = df[df['column_position'] == col]['shoulder_width'].values
        if len(data) > 0:
            plt.hist(data, bins=bins, alpha=0.4, label=f'列 {col}', color=colors[i], edgecolor='black')

    plt.xticks(bins)
    plt.grid(True, alpha=0.3)
    for x in bins:
        plt.axvline(x, color='gray', linestyle='dotted', linewidth=0.7, alpha=0.5)

    plt.xlabel('肩幅 (px)', fontsize=13)
    plt.ylabel('度数', fontsize=13)
    plt.title(f'各列ごとの肩幅分布（ヒストグラム）{"（IQR除去）" if suffix == "_iqr" else ""}', fontsize=15)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"shoulder_width_hist{suffix}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 各列ごとの肩幅分布（ヒストグラム）を {output_path} に保存しました")



def main():
    print("🔧 距離正規化関数分析ツール（person_id選別・除外ID指定・2種グラフ・MSE計算・KDE/ヒストグラム分布可視化）")
    filtered_df = select_valid_ids_interactive()
    if 'column_position' not in filtered_df.columns or 'shoulder_width' not in filtered_df.columns:
        print("❌ データCSVに必要なカラムがありません（column_position, shoulder_width）")
        sys.exit(1)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"analysis_{timestamp}"
    (fit_params_exp_all, fit_params_lin_all,
     fit_params_exp_iqr, fit_params_lin_iqr,
     df_all, df_iqr) = plot_shoulder_width_vs_column_with_fit(filtered_df, output_dir)
    # MSE計算（全データ）
    calc_and_save_mse(df_all, fit_params_exp_all, fit_params_lin_all, output_dir, suffix="_all")
    # MSE計算（IQR除去データ）
    calc_and_save_mse(df_iqr, fit_params_exp_iqr, fit_params_lin_iqr, output_dir, suffix="_iqr")
    # 各列ごとの分布（KDE, 全データ）
    plot_kde_by_column(filtered_df, output_dir, suffix="")
    # 各列ごとの分布（ヒストグラム, 全データ）
    plot_hist_by_column(filtered_df, output_dir, suffix="")
    # 各列ごとの分布（KDE, IQR除去データ）
    plot_kde_by_column(df_iqr, output_dir, suffix="_iqr")
    # 各列ごとの分布（ヒストグラム, IQR除去データ）
    plot_hist_by_column(df_iqr, output_dir, suffix="_iqr")

if __name__ == "__main__":
    main()