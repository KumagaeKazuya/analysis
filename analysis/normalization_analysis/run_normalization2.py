#!/usr/bin/env python3
"""
🔧 距離正規化関数分析 + 評価指標統合版

改善点:
1. 課題1評価: 列間比率・相関係数・変動係数の自動計算
2. 課題2評価: 検出率・信頼度分布・消失率の自動計算  
3. 課題3評価: 分散比・ICC・個人内外変動係数の自動計算
4. 正規化前後の評価指標比較レポート生成
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
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'AppleGothic'

# ============================================================================
# 📊 評価指標計算関数群
# ============================================================================

def evaluate_task1_scale_variation(df, column_col='column_position', 
                                   shoulder_col='shoulder_width'):
    """
    課題1: カメラ距離によるスケール変化の評価
    
    Returns:
        dict: {
            'column_ratios': 列間比率 (w1/w2, w2/w3),
            'correlation': Pearson相関係数,
            'cv_between_columns': 列間変動係数
        }
    """
    results = {}
    
    # 列ごとの平均肩幅
    column_means = df.groupby(column_col)[shoulder_col].mean().sort_index()
    columns = column_means.index.tolist()
    
    # 1. 列間比率の計算
    ratios = {}
    for i in range(len(columns)-1):
        ratio = column_means.iloc[i] / column_means.iloc[i+1]
        ratios[f'w{columns[i]}/w{columns[i+1]}'] = ratio
    results['column_ratios'] = ratios
    
    # 2. Pearson相関係数（列位置 vs 肩幅）
    corr, p_value = pearsonr(df[column_col], df[shoulder_col])
    results['correlation'] = {'r': corr, 'p_value': p_value}
    
    # 3. 列間変動係数 (CV_between)
    sigma_between = column_means.std()
    mu_overall = column_means.mean()
    cv_between = sigma_between / mu_overall if mu_overall > 0 else 0
    results['cv_between_columns'] = cv_between
    
    return results


def evaluate_task2_detection_stability(df, column_col='column_position',
                                       conf_col='left_shoulder_conf',
                                       frame_col='frame', id_col='person_id',
                                       conf_threshold=0.5):
    """
    課題2: 遮蔽による検出不安定性の評価
    
    Returns:
        dict: {
            'detection_rates': 列別検出率,
            'confidence_stats': 信頼度統計,
            'disappearance_rates': 消失率
        }
    """
    results = {}
    
    # 1. 列別検出率
    detection_rates = {}
    for col in sorted(df[column_col].unique()):
        df_col = df[df[column_col] == col]
        total_frames = len(df_col)
        detected_frames = len(df_col[df_col[conf_col] >= conf_threshold])
        rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
        detection_rates[f'column_{col}'] = rate
    results['detection_rates'] = detection_rates
    
    # 2. 信頼度分布統計
    conf_stats = {}
    for col in sorted(df[column_col].unique()):
        df_col = df[df[column_col] == col]
        conf_stats[f'column_{col}'] = {
            'median': df_col[conf_col].median(),
            'q1': df_col[conf_col].quantile(0.25),
            'mean': df_col[conf_col].mean(),
            'std': df_col[conf_col].std()
        }
    results['confidence_stats'] = conf_stats
    
    # 3. 消失率（40フレーム間隔でのID追跡継続率）
    disappearance_rates = {}
    for col in sorted(df[column_col].unique()):
        df_col = df[df[column_col] == col].copy()
        df_col['frame_num'] = df_col[frame_col].astype(str).str.extract(r'(\d+)')[0].astype(int)
        
        total_pairs = 0
        loss_pairs = 0
        
        for pid in df_col[id_col].unique():
            df_pid = df_col[df_col[id_col] == pid].sort_values('frame_num')
            frames = df_pid['frame_num'].values
            
            for i in range(len(frames)-1):
                if frames[i+1] - frames[i] >= 40:
                    total_pairs += 1
                    # 次のフレームで検出されなければ消失
                    next_frame = frames[i] + 40
                    if next_frame not in frames:
                        loss_pairs += 1
        
        rate = (loss_pairs / total_pairs * 100) if total_pairs > 0 else 0
        disappearance_rates[f'column_{col}'] = rate
    results['disappearance_rates'] = disappearance_rates
    
    return results


def evaluate_task3_individual_variation(df, column_col='column_position',
                                        shoulder_col='shoulder_width',
                                        id_col='person_id'):
    """
    課題3: 個人差の影響評価
    
    Returns:
        dict: {
            'variance_ratios': 列別の個人間/個人内分散比,
            'icc': 級内相関係数,
            'cv_between_within': 個人間・個人内変動係数
        }
    """
    results = {}
    
    # 列ごとに評価
    for col in sorted(df[column_col].unique()):
        df_col = df[df[column_col] == col].copy()
        
        # 1. 個人間分散 vs 個人内分散
        person_means = df_col.groupby(id_col)[shoulder_col].mean()
        column_mean = df_col[shoulder_col].mean()
        
        # 個人間分散
        sigma2_between = ((person_means - column_mean) ** 2).sum() / (len(person_means) - 1)
        
        # 個人内分散の平均
        within_vars = []
        for pid in df_col[id_col].unique():
            df_pid = df_col[df_col[id_col] == pid]
            if len(df_pid) > 1:
                within_var = df_pid[shoulder_col].var()
                within_vars.append(within_var)
        sigma2_within = np.mean(within_vars) if within_vars else 0
        
        # 分散比
        variance_ratio = sigma2_between / sigma2_within if sigma2_within > 0 else np.inf
        
        # 2. ICC(2,1)
        icc = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0
        
        # 3. 変動係数
        cv_between = np.sqrt(sigma2_between) / column_mean if column_mean > 0 else 0
        cv_within = np.sqrt(sigma2_within) / column_mean if column_mean > 0 else 0
        
        results[f'column_{col}'] = {
            'variance_ratio': variance_ratio,
            'icc': icc,
            'cv_between': cv_between,
            'cv_within': cv_within
        }
    
    return results


def generate_evaluation_report(df_before, df_after, output_dir):
    """
    正規化前後の評価指標比較レポート生成
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'before_normalization': {},
        'after_normalization': {},
        'improvements': {}
    }
    
    # 正規化前
    print("\n" + "="*60)
    print("📊 正規化前の評価指標")
    print("="*60)
    
    task1_before = evaluate_task1_scale_variation(df_before)
    task2_before = evaluate_task2_detection_stability(df_before)
    task3_before = evaluate_task3_individual_variation(df_before)
    
    report['before_normalization'] = {
        'task1_scale': task1_before,
        'task2_stability': task2_before,
        'task3_individual': task3_before
    }
    
    # 結果表示
    print("\n【課題1: スケール変化】")
    print(f"  列間比率: {task1_before['column_ratios']}")
    print(f"  相関係数: r={task1_before['correlation']['r']:.3f}, p={task1_before['correlation']['p_value']:.4f}")
    print(f"  列間CV: {task1_before['cv_between_columns']:.3f}")
    
    print("\n【課題2: 検出安定性】")
    print(f"  検出率: {task2_before['detection_rates']}")
    print(f"  消失率: {task2_before['disappearance_rates']}")
    
    print("\n【課題3: 個人差】")
    for col, stats in task3_before.items():
        print(f"  {col}: 分散比={stats['variance_ratio']:.2f}, ICC={stats['icc']:.3f}")
    
    # 正規化後
    if df_after is not None:
        print("\n" + "="*60)
        print("📊 正規化後の評価指標")
        print("="*60)
        
        task1_after = evaluate_task1_scale_variation(df_after, shoulder_col='shoulder_width_pred')
        
        report['after_normalization'] = {
            'task1_scale': task1_after
        }
        
        print("\n【課題1: スケール変化】")
        print(f"  列間比率: {task1_after['column_ratios']}")
        print(f"  相関係数: r={task1_after['correlation']['r']:.3f}, p={task1_after['correlation']['p_value']:.4f}")
        print(f"  列間CV: {task1_after['cv_between_columns']:.3f}")
        
        # 改善度計算
        cv_improvement = ((task1_before['cv_between_columns'] - task1_after['cv_between_columns']) 
                         / task1_before['cv_between_columns'] * 100)
        
        report['improvements'] = {
            'cv_reduction_percent': cv_improvement,
            'target_achieved': cv_improvement >= 50  # 目標: 50%以上削減
        }
        
        print("\n" + "="*60)
        print("📈 改善度評価")
        print("="*60)
        print(f"  列間CV削減率: {cv_improvement:.1f}%")
        print(f"  目標達成: {'✅ Yes' if cv_improvement >= 50 else '❌ No (目標50%以上)'}")
    
    # レポート保存
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 評価レポート保存: {report_path}")
    
    return report


# ============================================================================
# 既存関数（変更なし）
# ============================================================================

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


def save_median_rows_per_minute(df, id_col='person_id', frame_col='frame', 
                                shoulder_col='shoulder_width',
                                out_dir='median_rows_by_minute', 
                                frames_per_minute=1200):
    # 既存実装そのまま
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


def aggregate_by_column_position(median_dir, out_dir):
    # 既存実装そのまま
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


def fit_mixed_effect_and_save_params_from_csv(column_dir, output_dir):
    # 既存実装そのまま + 正規化後データフレーム返却
    files = glob.glob(os.path.join(column_dir, "column_*_median_rows.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'column_position' not in df.columns or 'person_id' not in df.columns or 'shoulder_width' not in df.columns:
            continue
        dfs.append(df[['column_position', 'person_id', 'shoulder_width']])
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['column_position'] = df_all['column_position'].astype(int)
    
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
    
    save_params(fit_params_exp, "exp_mixed", "f(x) = a * exp(-b * x) + c")
    save_params(fit_params_lin, "linear_mixed", "f(x) = a * x + b + c")
    
    # グラフ描画
    plt.figure(figsize=(10, 7))
    plt.scatter(pred_df['column_position'], pred_df['shoulder_width_pred'], 
               color='red', marker='D', s=80, label='階層回帰推定値')
    x_fit = np.linspace(xdata.min(), xdata.max(), 100)
    if fit_params_exp is not None:
        plt.plot(x_fit, exp_decay(x_fit, *fit_params_exp), color='blue', 
                linewidth=2, label='指数減衰フィット')
    if fit_params_lin is not None:
        plt.plot(x_fit, fit_params_lin[0]*x_fit + fit_params_lin[1] + fit_params_lin[2], 
                color='green', linestyle='dashed', linewidth=2, label='直線フィット')
    plt.xlabel('列位置 (column_position)', fontsize=13)
    plt.ylabel('肩幅 (px)', fontsize=13)
    plt.title('距離-肩幅関係（階層回帰推定値・近似）', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"shoulder_width_vs_column_fit_mixed.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return fit_params_exp, fit_params_lin, pred_df


def apply_normalization_to_full_data(filtered_df, fit_params_lin, fit_params_exp):
    """
    全データに正規化を適用（個人別・フレーム別）
    """
    df_normalized = filtered_df.copy()
    
    # 線形正規化
    if fit_params_lin is not None:
        a, b, c = fit_params_lin
        def normalize_linear(row):
            col_pos = row['column_position']
            original = row['shoulder_width']
            expected = a * col_pos + b + c
            # 基準列（例: 列1）の期待値
            expected_ref = a * 1 + b + c
            return original * (expected_ref / expected) if expected > 0 else original
        
        df_normalized['shoulder_width_normalized_linear'] = df_normalized.apply(
            normalize_linear, axis=1
        )
    
    # 指数正規化
    if fit_params_exp is not None:
        a, b, c = fit_params_exp
        def normalize_exp(row):
            col_pos = row['column_position']
            original = row['shoulder_width']
            expected = a * np.exp(-b * col_pos) + c
            expected_ref = a * np.exp(-b * 1) + c
            return original * (expected_ref / expected) if expected > 0 else original
        
        df_normalized['shoulder_width_normalized_exp'] = df_normalized.apply(
            normalize_exp, axis=1
        )
    
    return df_normalized


def main():
    """
    メイン実行関数（評価指標統合版・完全版）
    """
    print("🔧 距離正規化関数分析ツール + 評価指標統合版")
    
    # データ選択
    filtered_df = select_valid_ids_interactive()
    
    # column_position確認
    if 'column_position' not in filtered_df.columns:
        print("\n⚠️ 警告: column_positionカラムが見つかりません")
        print("列位置を手動で入力してください（例: 1, 2, 3）")
        try:
            filtered_df['column_position'] = int(input("列位置 > "))
            print(f"✅ 全データを列位置 {filtered_df['column_position'].iloc[0]} として処理します")
        except:
            print("❌ 列位置の設定に失敗しました。処理を中断します。")
            sys.exit(1)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # ⭐ 正規化前の評価指標計算
    print("\n" + "="*60)
    print("📊 Step 1: 正規化前の評価指標計算")
    print("="*60)
    
    # 必須カラム確認
    required_cols = ['column_position', 'shoulder_width', 'person_id', 'frame']
    missing_cols = [col for col in required_cols if col not in filtered_df.columns]
    if missing_cols:
        print(f"❌ エラー: 必須カラムが不足しています: {missing_cols}")
        print(f"   利用可能なカラム: {list(filtered_df.columns)}")
        sys.exit(1)
    
    try:
        task1_before = evaluate_task1_scale_variation(filtered_df)
        task2_before = evaluate_task2_detection_stability(filtered_df)
        task3_before = evaluate_task3_individual_variation(filtered_df)
        
        print("\n【課題1: スケール変化】")
        print(f"  列間比率: {task1_before['column_ratios']}")
        print(f"  相関係数: r={task1_before['correlation']['r']:.3f}")
        print(f"  列間CV: {task1_before['cv_between_columns']:.3f}")
        
        print("\n【課題2: 検出安定性】")
        print(f"  検出率: {task2_before['detection_rates']}")
        
        print("\n【課題3: 個人差】")
        for col, stats in task3_before.items():
            print(f"  {col}: ICC={stats['icc']:.3f}, 分散比={stats['variance_ratio']:.2f}")
    except Exception as e:
        print(f"⚠️ 評価指標計算エラー（処理は継続）: {e}")
        task1_before = task2_before = task3_before = None
    
    # 既存処理
    print("\n" + "="*60)
    print("📊 Step 2: 階層回帰モデル構築")
    print("="*60)
    
    median_dir = os.path.join(output_dir, 'median_rows_by_minute')
    save_median_rows_per_minute(filtered_df, out_dir=median_dir)
    
    column_dir = os.path.join(output_dir, 'column_median_rows')
    aggregate_by_column_position(median_dir, column_dir)
    
    fit_params_exp, fit_params_lin, pred_df = fit_mixed_effect_and_save_params_from_csv(
        column_dir, output_dir
    )
    
    # ⭐ 全データに正規化適用
    print("\n" + "="*60)
    print("📊 Step 3: 全データへの正規化適用")
    print("="*60)
    
    df_normalized = apply_normalization_to_full_data(
        filtered_df, fit_params_lin, fit_params_exp
    )
    
    # 正規化後データ保存
    normalized_csv = os.path.join(output_dir, "normalized_full_data.csv")
    df_normalized.to_csv(normalized_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 正規化済み全データ保存: {normalized_csv}")
    
    # ⭐ 正規化後の評価指標計算
    print("\n" + "="*60)
    print("📊 Step 4: 正規化後の評価指標計算")
    print("="*60)
    
    try:
        # 線形正規化の評価
        if 'shoulder_width_normalized_linear' in df_normalized.columns:
            task1_after_linear = evaluate_task1_scale_variation(
                df_normalized, shoulder_col='shoulder_width_normalized_linear'
            )
            print("\n【線形正規化後】")
            print(f"  列間CV: {task1_after_linear['cv_between_columns']:.3f}")
            
            cv_improvement_linear = (
                (task1_before['cv_between_columns'] - task1_after_linear['cv_between_columns'])
                / task1_before['cv_between_columns'] * 100
            )
            print(f"  CV削減率: {cv_improvement_linear:.1f}%")
            print(f"  目標達成: {'✅' if cv_improvement_linear >= 50 else '❌'} (目標50%以上)")
        
        # 指数正規化の評価
        if 'shoulder_width_normalized_exp' in df_normalized.columns:
            task1_after_exp = evaluate_task1_scale_variation(
                df_normalized, shoulder_col='shoulder_width_normalized_exp'
            )
            print("\n【指数正規化後】")
            print(f"  列間CV: {task1_after_exp['cv_between_columns']:.3f}")
            
            cv_improvement_exp = (
                (task1_before['cv_between_columns'] - task1_after_exp['cv_between_columns'])
                / task1_before['cv_between_columns'] * 100
            )
            print(f"  CV削減率: {cv_improvement_exp:.1f}%")
            print(f"  目標達成: {'✅' if cv_improvement_exp >= 50 else '❌'} (目標50%以上)")
    
    except Exception as e:
        print(f"⚠️ 正規化後評価エラー: {e}")
    
    # 統合レポート生成
    print("\n" + "="*60)
    print("📊 Step 5: 統合レポート生成")
    print("="*60)
    
    report = {
        'timestamp': timestamp,
        'before_normalization': {
            'task1': task1_before if task1_before else {},
            'task2': task2_before if task2_before else {},
            'task3': task3_before if task3_before else {}
        },
        'after_normalization': {},
        'improvements': {}
    }
    
    if task1_before and 'shoulder_width_normalized_linear' in df_normalized.columns:
        report['after_normalization']['linear'] = task1_after_linear
        report['improvements']['linear_cv_reduction'] = cv_improvement_linear
    
    if task1_before and 'shoulder_width_normalized_exp' in df_normalized.columns:
        report['after_normalization']['exp'] = task1_after_exp
        report['improvements']['exp_cv_reduction'] = cv_improvement_exp
    
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✅ 全処理完了!")
    print("="*80)
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"📄 正規化データ: {normalized_csv}")
    print(f"📄 評価レポート: {report_path}")


if __name__ == "__main__":
    main()