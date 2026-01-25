#!/usr/bin/env python3
"""
階層回帰モデルによる正規化関数獲得と評価
論文記載の計算内容のみを実装
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import shutil
import glob
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'AppleGothic'

# ============================================================================
# 論文記載の評価指標計算関数
# ============================================================================

def calculate_task1_metrics(df, column_col='column_position', 
                            shoulder_col='shoulder_width'):
    """
    課題1: カメラ距離によるスケール変化の評価
    論文 表5.6, 表5.7に対応
    
    計算項目:
    1. 列間比率 R_{1,2} = w̄_1 / w̄_2
    2. Pearson相関係数 r (列位置 vs 肩幅)
    3. 列間変動係数 CV_between
    """
    results = {}
    
    # 各個人のID別平均値を計算
    person_stats = df.groupby(['person_id', column_col])[shoulder_col].agg(['mean', 'std', 'count']).reset_index()
    
    # 列ごとの平均値（個人平均の平均）
    column_means = person_stats.groupby(column_col)['mean'].mean().sort_index()
    
    print(f"\n列ごとの平均肩幅:")
    for col, mean_val in column_means.items():
        print(f"  第{col}列: {mean_val:.1f} px")
    
    # 1. 列間比率の計算
    columns = column_means.index.tolist()
    if len(columns) >= 2:
        R_12 = column_means.iloc[0] / column_means.iloc[1]
        results['column_ratio_R12'] = R_12
        print(f"\n列間比率 R_{{1,2}}: {R_12:.3f}")
    
    # 2. Pearson相関係数（列位置 vs 肩幅平均値）
    # 論文では列ごとの平均値との相関を計算
    if len(columns) >= 2:
        corr, p_value = pearsonr(columns, column_means.values)
        results['correlation_r'] = corr
        results['correlation_p'] = p_value
        print(f"相関係数 r: {corr:.3f} (p={p_value:.4f})")
    
    # 3. 列間変動係数 CV_between
    sigma_between = column_means.std()
    mu_overall = column_means.mean()
    cv_between = (sigma_between / mu_overall) * 100 if mu_overall > 0 else 0
    results['cv_between_columns'] = cv_between
    print(f"列間変動係数 CV: {cv_between:.1f}%")
    
    return results, person_stats


def calculate_individual_cv(df, id_col='person_id', shoulder_col='shoulder_width'):
    """
    個人内変動係数 CV_within の計算
    論文 表5.5に対応
    """
    cv_results = []
    
    for pid in df[id_col].unique():
        df_pid = df[df[id_col] == pid]
        mean_val = df_pid[shoulder_col].mean()
        std_val = df_pid[shoulder_col].std()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        
        cv_results.append({
            'person_id': pid,
            'mean': mean_val,
            'std': std_val,
            'cv_within': cv,
            'count': len(df_pid)
        })
    
    return pd.DataFrame(cv_results)


# ============================================================================
# データ選別・フィルタリング
# ============================================================================

def select_valid_ids_with_filters():
    """
    論文のフィルタリング条件に従ってIDを選別
    
    第1層: 信頼度0.5未満のフレームを除外（事前処理済み）
    第2層: 全体取得枚数の50%以上
    第3層: 1分間の時間窓で有効データが20%以上
    """
    print("=" * 60)
    print("フィルタリング済みデータの読み込み")
    print("=" * 60)
    
    # 第2層フィルタリング用のサマリーCSV
    print("\n【第2層フィルタリング】全体取得枚数の50%以上")
    id_csv_path = input("サマリーCSVパス (例: summary.csv): ").strip()
    
    if not os.path.exists(id_csv_path):
        print(f"❌ ファイルが見つかりません: {id_csv_path}")
        sys.exit(1)
    
    id_df = pd.read_csv(id_csv_path)
    
    # 50%以上のIDを抽出
    if 'frames_with_id' in id_df.columns and 'total_frames' in id_df.columns:
        id_df['detection_rate'] = id_df['frames_with_id'] / id_df['total_frames']
        valid_ids_layer2 = id_df.loc[id_df['detection_rate'] >= 0.5, 'person_id'].tolist()
        print(f"第2層通過ID: {valid_ids_layer2}")
    else:
        print("❌ 必要なカラムがありません")
        sys.exit(1)
    
    # 第3層フィルタリング用の除外ID指定
    print("\n【第3層フィルタリング】時間窓検出率20%未満のID除外")
    print("除外するIDをカンマ区切りで入力してください（例: 2,3,19）")
    print("除外しない場合は何も入力せずEnter")
    exclude_input = input("除外ID: ").strip()
    
    exclude_ids = []
    if exclude_input:
        try:
            exclude_ids = [int(x.strip()) for x in exclude_input.split(",")]
            print(f"除外ID: {exclude_ids}")
        except:
            print("⚠️ 入力形式エラー。除外IDなしで続行します。")
    
    # 最終的な有効ID
    valid_ids = [pid for pid in valid_ids_layer2 if pid not in exclude_ids]
    print(f"\n最終有効ID: {valid_ids}")
    
    if not valid_ids:
        print("❌ 有効なIDがありません")
        sys.exit(1)
    
    # データ読み込み
    data_csv_path = input("\nデータCSVパス (例: filtered_data.csv): ").strip()
    
    if not os.path.exists(data_csv_path):
        print(f"❌ ファイルが見つかりません: {data_csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_csv_path)
    filtered_df = df[df['person_id'].isin(valid_ids)].copy()
    
    print(f"\n抽出データ件数: {len(filtered_df)} フレーム")
    print(f"対象ID数: {len(valid_ids)} 名")
    
    return filtered_df


# ============================================================================
# 時間窓による集約（論文 5.3.1項）
# ============================================================================

def aggregate_by_time_window(df, frames_per_window=1200):
    """
    1200フレーム（60秒）ごとに中央値を算出
    論文のリスト5.1 手順4に対応
    """
    print(f"\n時間窓集約: {frames_per_window}フレーム区間ごとに中央値を算出")
    
    aggregated_rows = []
    
    for pid in df['person_id'].unique():
        df_pid = df[df['person_id'] == pid].copy()
        
        # フレーム番号を数値化
        df_pid['frame_num'] = df_pid['frame'].astype(str).str.extract(r'(\d+)')[0].astype(int)
        df_pid = df_pid.sort_values('frame_num')
        
        frame_max = df_pid['frame_num'].max()
        frame_min = df_pid['frame_num'].min()
        
        # 時間窓の開始点
        window_starts = np.arange(frame_min, frame_max, frames_per_window)
        
        for start in window_starts:
            end = start + frames_per_window - 1
            
            # 窓内のデータ
            df_window = df_pid[(df_pid['frame_num'] >= start) & 
                              (df_pid['frame_num'] <= end)]
            
            if len(df_window) == 0:
                continue
            
            # 中央値を代表値として採用
            median_width = df_window['shoulder_width'].median()
            
            # 中央値に最も近いフレームを選択
            median_idx = (df_window['shoulder_width'] - median_width).abs().idxmin()
            median_row = df_window.loc[median_idx].copy()
            median_row['window_start'] = start
            median_row['window_end'] = end
            
            aggregated_rows.append(median_row)
    
    result_df = pd.DataFrame(aggregated_rows)
    print(f"集約後データ件数: {len(result_df)}")
    
    return result_df


# ============================================================================
# 階層回帰モデル（論文 式5.24）
# ============================================================================

def fit_hierarchical_regression(df):
    """
    線形混合効果モデルによる階層回帰
    
    モデル式: y_ij = β_0 + β_1 * I(j=2) + u_i + ε_ij
    """
    print("\n" + "=" * 60)
    print("階層回帰モデル構築")
    print("=" * 60)
    
    required = ['column_position', 'person_id', 'shoulder_width']
    if not all(col in df.columns for col in required):
        print(f"❌ 必要なカラムが不足: {required}")
        sys.exit(1)
    
    df['column_position'] = df['column_position'].astype(int)
    
    print("\nモデル: shoulder_width ~ C(column_position)")
    print("ランダム効果: person_id")
    print("推定方法: REML (Restricted Maximum Likelihood)")
    
    model = smf.mixedlm(
        "shoulder_width ~ C(column_position)", 
        df, 
        groups=df["person_id"]
    )
    result = model.fit(reml=True)  # REML推定を明示
    
    print("\n" + result.summary().as_text())
    
    # 推定結果の詳細を取得
    model_info = {
        'converged': result.converged,
        'iterations': result.method if hasattr(result, 'method') else 'N/A',
        'loglikelihood': result.llf,
        'aic': result.aic,
        'bic': result.bic,
        
        # 固定効果
        'fixed_effects': {
            'intercept': {
                'estimate': result.params['Intercept'],
                'se': result.bse['Intercept'],
                'tvalue': result.tvalues['Intercept'],
                'pvalue': result.pvalues['Intercept']
            }
        },
        
        # 分散成分
        'random_effects_var': result.cov_re.iloc[0, 0],  # σ_u^2
        'residual_var': result.scale,  # σ_ε^2
    }
    
    # 列2の効果（存在する場合）
    col2_key = 'C(column_position)[T.2]'
    if col2_key in result.params:
        model_info['fixed_effects']['column2_effect'] = {
            'estimate': result.params[col2_key],
            'se': result.bse[col2_key],
            'tvalue': result.tvalues[col2_key],
            'pvalue': result.pvalues[col2_key]
        }
    
    # ICC計算
    var_u = model_info['random_effects_var']
    var_e = model_info['residual_var']
    icc = var_u / (var_u + var_e)
    model_info['icc'] = icc
    
    print(f"\n【分散成分】")
    print(f"  個人間分散 σ_u^2: {var_u:.2f} (SD: {np.sqrt(var_u):.2f})")
    print(f"  個人内分散 σ_ε^2: {var_e:.2f} (SD: {np.sqrt(var_e):.2f})")
    print(f"  級内相関係数 ICC: {icc:.3f}")
    
    # 各列の推定値を抽出
    columns = sorted(df['column_position'].unique())
    estimates = {}
    
    intercept = result.params['Intercept']
    estimates[columns[0]] = intercept
    
    for col in columns[1:]:
        key = f'C(column_position)[T.{col}]'
        if key in result.params:
            estimates[col] = intercept + result.params[key]
        else:
            estimates[col] = intercept
    
    print(f"\n【列ごとの母集団平均推定値】")
    for col, est in estimates.items():
        print(f"  第{col}列: {est:.2f} px")
    
    return estimates, result, model_info


# ============================================================================
# 正規化関数の獲得（論文 直線近似）
# ============================================================================

def fit_normalization_function(estimates):
    """
    階層回帰の推定値に対して直線近似
    f(x) = ax + b
    """
    print("\n" + "=" * 60)
    print("正規化関数の獲得（直線近似）")
    print("=" * 60)
    
    columns = np.array(list(estimates.keys()))
    widths = np.array(list(estimates.values()))
    
    # 1次多項式フィッティング
    coeffs = np.polyfit(columns, widths, 1)
    a, b = coeffs
    
    print(f"\n正規化関数: f(x) = {a:.2f}x + {b:.2f}")
    print(f"  傾き a = {a:.2f}")
    print(f"  切片 b = {b:.2f}")
    
    # グラフ描画
    plt.figure(figsize=(10, 7))
    plt.scatter(columns, widths, color='red', marker='D', s=100, 
               label='階層回帰推定値', zorder=3)
    
    x_line = np.linspace(columns.min(), columns.max(), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, 'b-', linewidth=2, label=f'f(x) = {a:.2f}x + {b:.2f}')
    
    plt.xlabel('列位置 (column_position)', fontsize=13)
    plt.ylabel('肩幅 (px)', fontsize=13)
    plt.title('階層回帰推定値に対するフィッティング結果', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {'a': a, 'b': b}, plt.gcf()


# ============================================================================
# 正規化の適用
# ============================================================================

def apply_normalization(df, params):
    """
    正規化関数を全データに適用
    
    論文の記述: 「各データの観測肩幅を、列位置から算出した期待肩幅で除算」
    normalized = original / f(column_position)
    """
    print("\n" + "=" * 60)
    print("全データへの正規化適用")
    print("=" * 60)
    
    df_norm = df.copy()
    a, b = params['a'], params['b']
    
    def normalize_width(row):
        col_pos = row['column_position']
        original = row['shoulder_width']
        expected = a * col_pos + b
        
        # 期待値で除算（論文の記述通り）
        if expected > 0:
            return original / expected
        else:
            return original
    
    df_norm['shoulder_width_normalized'] = df_norm.apply(normalize_width, axis=1)
    
    print(f"正規化完了: {len(df_norm)} フレーム")
    
    return df_norm


# ============================================================================
# メイン処理
# ============================================================================

def main():
    """
    論文記載の処理手順に従った実行
    
    リスト5.2の処理手順:
    1. 出力フォルダ作成
    2. person_id選別
    3. データ抽出
    4. 正規化前評価指標計算
    5. 階層回帰モデル構築
    6. 直線近似・パラメータ保存
    7. 全データ正規化・保存
    8. 正規化後評価指標計算
    9. レポート生成・保存
    """
    print("=" * 60)
    print("階層回帰モデルによる正規化関数獲得")
    print("論文準拠版")
    print("=" * 60)
    
    # 1. 出力フォルダ作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"normalization_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n出力ディレクトリ: {output_dir}")
    
    # 2-3. person_id選別とデータ抽出
    df = select_valid_ids_with_filters()
    
    # データ保存
    df.to_csv(os.path.join(output_dir, 'filtered_data.csv'), 
             index=False, encoding='utf-8-sig')
    
    # 4. 正規化前評価指標計算
    print("\n" + "=" * 60)
    print("正規化前の評価指標")
    print("=" * 60)
    
    # 個人内変動係数
    cv_df = calculate_individual_cv(df)
    cv_df.to_csv(os.path.join(output_dir, 'cv_within_before.csv'), 
                index=False, encoding='utf-8-sig')
    
    print("\n個人別統計量:")
    print(cv_df.to_string(index=False))
    
    # 課題1評価指標
    task1_before, person_stats = calculate_task1_metrics(df)
    
    # 5. 階層回帰モデル構築（時間窓集約後）
    df_aggregated = aggregate_by_time_window(df)
    estimates, model_result, model_info = fit_hierarchical_regression(df_aggregated)

    # model_info を保存
    with open(os.path.join(output_dir, 'model_details.json'), 
             'w', encoding='utf-8') as f:
        # NumPy型をPython標準型に変換
        model_info_serializable = {
            k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
            for k, v in model_info.items()
        }
        json.dump(model_info_serializable, f, indent=2, ensure_ascii=False)
    
    # 6. 直線近似・パラメータ保存
    params, fig = fit_normalization_function(estimates)
    
    # パラメータ保存
    with open(os.path.join(output_dir, 'normalization_params.json'), 
             'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    # グラフ保存
    fig.savefig(os.path.join(output_dir, 'fitting_result.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 全データ正規化・保存
    df_normalized = apply_normalization(df, params)
    df_normalized.to_csv(os.path.join(output_dir, 'normalized_data.csv'),
                        index=False, encoding='utf-8-sig')
    
    # 8. 正規化後評価指標計算
    print("\n" + "=" * 60)
    print("正規化後の評価指標")
    print("=" * 60)
    
    # 正規化後の課題1評価
    task1_after, _ = calculate_task1_metrics(
        df_normalized, 
        shoulder_col='shoulder_width_normalized'
    )
    
    # 正規化後の個人内変動係数
    cv_df_after = calculate_individual_cv(
        df_normalized, 
        shoulder_col='shoulder_width_normalized'
    )
    cv_df_after.to_csv(os.path.join(output_dir, 'cv_within_after.csv'),
                      index=False, encoding='utf-8-sig')
    
    # 9. レポート生成・保存
    print("\n" + "=" * 60)
    print("改善度評価（表5.8対応）")
    print("=" * 60)
    
    report = {
        'timestamp': timestamp,
        'normalization_params': params,
        'before_normalization': task1_before,
        'after_normalization': task1_after,
        'improvements': {}
    }
    
    # 改善度計算
    if 'column_ratio_R12' in task1_before and 'column_ratio_R12' in task1_after:
        report['improvements']['ratio_change'] = {
            'before': task1_before['column_ratio_R12'],
            'after': task1_after['column_ratio_R12'],
            'target': 1.000
        }
        print(f"列間比率 R_{{1,2}}:")
        print(f"  正規化前: {task1_before['column_ratio_R12']:.3f}")
        print(f"  正規化後: {task1_after['column_ratio_R12']:.3f}")
        print(f"  目標値: 1.000")
    
    if 'correlation_r' in task1_before and 'correlation_r' in task1_after:
        report['improvements']['correlation_change'] = {
            'before': task1_before['correlation_r'],
            'after': task1_after['correlation_r'],
            'target': 0.000
        }
        print(f"\n相関係数 r:")
        print(f"  正規化前: {task1_before['correlation_r']:.3f}")
        print(f"  正規化後: {task1_after['correlation_r']:.3f}")
        print(f"  目標値: 0.000")
    
    if 'cv_between_columns' in task1_before and 'cv_between_columns' in task1_after:
        cv_reduction = ((task1_before['cv_between_columns'] - 
                        task1_after['cv_between_columns']) / 
                       task1_before['cv_between_columns'] * 100)
        
        report['improvements']['cv_reduction'] = {
            'before': task1_before['cv_between_columns'],
            'after': task1_after['cv_between_columns'],
            'reduction_percent': cv_reduction
        }
        print(f"\n列間変動係数 CV:")
        print(f"  正規化前: {task1_before['cv_between_columns']:.1f}%")
        print(f"  正規化後: {task1_after['cv_between_columns']:.1f}%")
        print(f"  削減率: {cv_reduction:.1f}%")
    
    # レポート保存
    with open(os.path.join(output_dir, 'evaluation_report.json'),
             'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✅ 全処理完了")
    print("=" * 60)
    print(f"📁 出力先: {output_dir}/")
    print(f"  - filtered_data.csv: フィルタリング済みデータ")
    print(f"  - normalization_params.json: 正規化関数パラメータ")
    print(f"  - fitting_result.png: フィッティング結果グラフ")
    print(f"  - normalized_data.csv: 正規化済みデータ")
    print(f"  - evaluation_report.json: 評価レポート")
    print(f"  - cv_within_before.csv: 正規化前個人内CV")
    print(f"  - cv_within_after.csv: 正規化後個人内CV")


if __name__ == "__main__":
    main()