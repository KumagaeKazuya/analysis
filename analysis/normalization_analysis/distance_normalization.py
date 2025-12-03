import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

class DistanceNormalizationAnalyzer:
    """距離正規化用減少関数分析クラス"""
    
    def __init__(self, csv_path: str, output_base_dir: str = "outputs/normalization_analysis"):
        """
        Args:
            csv_path: 4点メトリクスCSVファイルのパス
            output_base_dir: 結果保存ディレクトリ
        """
        self.csv_path = Path(csv_path)
        self.output_base_dir = Path(output_base_dir)
        
        # 出力ディレクトリ作成（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_base_dir / f"analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ読み込み
        self._load_data()
        
        print(f"✅ 距離正規化分析器初期化完了")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def _load_data(self):
        """CSVデータの読み込みと検証"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # 必要列の確認
        required_cols = ['frame', 'person_id']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"必要な列が不足: {missing}")
        
        # 肩幅列の特定（複数候補から自動選択）
        shoulder_cols = ['shoulder_width', 'shoulder_width_pixels', 'shoulder_distance']
        self.shoulder_col = None
        for col in shoulder_cols:
            if col in self.df.columns:
                self.shoulder_col = col
                break
        
        if self.shoulder_col is None:
            raise ValueError(f"肩幅データが見つかりません。利用可能列: {list(self.df.columns)}")
        
        print(f"📊 データ情報:")
        print(f"   総データ数: {len(self.df)}")
        print(f"   フレーム数: {self.df['frame'].nunique()}")
        print(f"   ID数: {self.df['person_id'].nunique()}")
        print(f"   肩幅列: {self.shoulder_col}")
    
    def analyze_distance_function(
        self, 
        frame_id: str,
        column_assignments: Dict[int, List[int]],
        width_range: Tuple[float, float] = (10, 500)
    ) -> Dict:
        """
        🎯 メイン分析メソッド: 距離減少関数の分析
        
        Args:
            frame_id: 分析対象フレーム (例: "11月12日 1.mp4_frame0.jpg")
            column_assignments: {列番号: [ID1, ID2, ...]} (例: {1: [1,2], 2: [3,4], 3: [5,6]})
            width_range: 有効肩幅範囲 (デフォルト: 10-500px)
            
        Returns:
            分析結果辞書（減少関数パラメータ、可視化パス、正規化コード含む）
        """
        print(f"\n🎯 === 距離正規化関数分析開始 ===")
        print(f"📅 分析時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎬 対象フレーム: {frame_id}")
        print(f"📋 列構成: {column_assignments}")
        
        try:
            # 🔍 Step 1: フレームデータ抽出
            frame_data = self._extract_frame_data(frame_id, width_range)
            if frame_data is None:
                return {'success': False, 'error': 'フレームデータ抽出失敗'}
            
            # 📊 Step 2: 列別データ整理
            column_data = self._organize_column_data(frame_data, column_assignments)
            
            # 📈 Step 3: 減少関数フィッティング
            function_results = self._fit_distance_functions(column_data)
            
            # 🎨 Step 4: 可視化作成
            visualization_path = self._create_normalization_visualization(
                column_data, function_results, frame_id
            )
            
            # 🔧 Step 5: 正規化関数コード生成
            normalization_code = self._generate_normalization_code(function_results)
            
            # 📝 Step 6: 詳細レポート作成
            report_path = self._save_analysis_report(
                frame_id, column_assignments, column_data, 
                function_results, normalization_code
            )
            
            # 📊 Step 7: 結果統合
            result = {
                'success': True,
                'analysis_info': {
                    'timestamp': datetime.now().isoformat(),
                    'frame_id': frame_id,
                    'column_assignments': column_assignments,
                    'output_dir': str(self.output_dir)
                },
                'column_statistics': {
                    col: {
                        'count': data['count'],
                        'mean_shoulder_width': data['mean_width'],
                        'std_shoulder_width': data['std_width']
                    } for col, data in column_data.items() if data['count'] > 0
                },
                'distance_functions': function_results,
                'normalization_code': normalization_code,
                'output_files': {
                    'visualization': visualization_path,
                    'report': report_path,
                    'function_data': str(self.output_dir / 'function_parameters.json')
                }
            }
            
            # JSON保存
            self._save_function_parameters(function_results)
            
            print(f"\n✅ === 分析完了 ===")
            if 'best_function' in function_results:
                print(f"📈 最適関数: {function_results['best_function']['name']}")
                print(f"🔗 相関係数: {function_results['correlation']['coefficient']:.3f}")
            print(f"📁 結果保存: {self.output_dir}")
            
            return result
            
        except Exception as e:
            print(f"❌ 分析エラー: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_frame_data(self, frame_id: str, width_range: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """🔍 フレームデータの抽出と有効性検証"""
        frame_data = self.df[self.df['frame'] == frame_id].copy()
        
        if frame_data.empty:
            available_frames = self.df['frame'].unique()[:10]
            print(f"❌ フレーム '{frame_id}' が見つかりません")
            print(f"📋 利用可能フレーム例: {list(available_frames)}")
            return None
        
        # 有効肩幅データフィルタリング
        min_width, max_width = width_range
        valid_mask = (
            (frame_data[self.shoulder_col] >= min_width) & 
            (frame_data[self.shoulder_col] <= max_width) &
            (frame_data[self.shoulder_col].notna())
        )
        
        frame_data = frame_data[valid_mask].copy()
        
        print(f"📊 フレームデータ:")
        print(f"   検出人数: {len(frame_data)}")
        print(f"   有効ID: {sorted(frame_data['person_id'].unique())}")
        print(f"   肩幅範囲: {frame_data[self.shoulder_col].min():.1f} - {frame_data[self.shoulder_col].max():.1f}px")
        
        return frame_data
    
    def _organize_column_data(self, frame_data: pd.DataFrame, column_assignments: Dict[int, List[int]]) -> Dict:
        """📊 列別データの整理（統計計算付き）"""
        column_data = {}
        
        for column_num, person_ids in column_assignments.items():
            # 指定IDでデータ抽出
            column_df = frame_data[frame_data['person_id'].isin(person_ids)]
            
            if not column_df.empty:
                shoulder_widths = column_df[self.shoulder_col].values
                column_data[column_num] = {
                    'assigned_ids': person_ids,
                    'found_ids': column_df['person_id'].tolist(),
                    'shoulder_widths': shoulder_widths.tolist(),
                    'mean_width': float(np.mean(shoulder_widths)),
                    'std_width': float(np.std(shoulder_widths)),
                    'min_width': float(np.min(shoulder_widths)),
                    'max_width': float(np.max(shoulder_widths)),
                    'count': len(shoulder_widths)
                }
                print(f"📋 {column_num}列目: {len(shoulder_widths)}人検出, 平均{np.mean(shoulder_widths):.1f}px")
                print(f"   割り当てID: {person_ids} → 検出ID: {column_df['person_id'].tolist()}")
            else:
                # データなしの列も記録
                column_data[column_num] = {
                    'assigned_ids': person_ids,
                    'found_ids': [],
                    'shoulder_widths': [],
                    'mean_width': 0,
                    'std_width': 0,
                    'min_width': 0,
                    'max_width': 0,
                    'count': 0
                }
                print(f"⚠️ {column_num}列目: 該当IDなし ({person_ids})")
        
        return column_data
    
    def _fit_distance_functions(self, column_data: Dict) -> Dict:
        """📈 複数の距離減少関数でフィッティング"""
        # 有効データのある列のみ使用
        valid_columns = {k: v for k, v in column_data.items() if v['count'] > 0}
        
        if len(valid_columns) < 2:
            return {'error': 'データ不足（2列以上必要）', 'valid_columns': len(valid_columns)}
        
        # データ準備
        x_data = np.array(list(valid_columns.keys()))  # 列番号 [1, 2, 3]
        y_data = np.array([valid_columns[col]['mean_width'] for col in x_data])  # 平均肩幅
        
        print(f"📊 フィッティングデータ:")
        print(f"   列位置(X): {x_data}")
        print(f"   平均肩幅(Y): {y_data}")
        
        # 基本相関分析
        correlation, p_value = pearsonr(x_data, y_data)
        print(f"🔗 基本相関: r={correlation:.3f}, p={p_value:.3f}")
        
        # 📈 複数の関数でフィッティング試行
        functions = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'name': '線形関数',
                'formula': 'f(x) = ax + b',
                'expected': 'right_down'  # 右肩下がり期待
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(-b * x) + c,
                'name': '指数減衰関数',
                'formula': 'f(x) = a*exp(-bx) + c',
                'expected': 'right_down'
            },
            'power': {
                'func': lambda x, a, b, c: a / (x ** b) + c,
                'name': 'べき関数',
                'formula': 'f(x) = a/x^b + c',
                'expected': 'right_down'
            },
            'polynomial2': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'name': '2次多項式',
                'formula': 'f(x) = ax² + bx + c',
                'expected': 'flexible'
            }
        }
        
        fitting_results = {}
        best_fit = None
        best_r2 = -np.inf
        
        for func_name, func_info in functions.items():
            try:
                # 📊 初期値設定（右肩下がりを想定）
                if func_name == 'linear':
                    # 線形: 負の傾きを期待
                    slope_estimate = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
                    p0 = [slope_estimate, y_data[0]]
                elif func_name == 'exponential':
                    # 指数減衰: 正の係数、正の減衰率
                    p0 = [y_data[0] - y_data[-1], 0.5, y_data[-1]]
                elif func_name == 'power':
                    # べき関数: 正のべき乗
                    p0 = [y_data[0] * x_data[0], 1, y_data[-1]]
                elif func_name == 'polynomial2':
                    # 2次多項式: 下に凸を想定
                    p0 = [-1, -5, y_data[0]]
                
                # 🔧 フィッティング実行
                popt, pcov = curve_fit(func_info['func'], x_data, y_data, p0=p0, maxfev=2000)
                
                # 📊 予測値計算
                y_pred = func_info['func'](x_data, *popt)
                
                # 📈 評価指標計算
                ss_res = np.sum((y_data - y_pred) ** 2)  # 残差平方和
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # 全平方和
                r2 = 1 - (ss_res / ss_tot)  # R²決定係数
                rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))  # RMSE
                
                # 📋 結果保存
                fitting_results[func_name] = {
                    'parameters': popt.tolist(),
                    'covariance': pcov.tolist() if pcov is not None else None,
                    'r2_score': float(r2),
                    'rmse': float(rmse),
                    'predicted_values': y_pred.tolist(),
                    'formula': func_info['formula'],
                    'name': func_info['name'],
                    'slope_direction': 'decreasing' if y_pred[0] > y_pred[-1] else 'increasing'
                }
                
                # 🏆 最適関数選択（R²値基準）
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = func_name
                
                direction = "↘️" if y_pred[0] > y_pred[-1] else "↗️"
                print(f"🔧 {func_info['name']}: R²={r2:.3f}, RMSE={rmse:.1f}, 傾向={direction}")
                
            except Exception as e:
                print(f"⚠️ {func_info['name']}フィッティング失敗: {e}")
                continue
        
        return {
            'correlation': {
                'coefficient': float(correlation),
                'p_value': float(p_value),
                'interpretation': self._interpret_correlation(correlation)
            },
            'fitting_results': fitting_results,
            'best_function': {
                'name': best_fit,
                'r2_score': float(best_r2) if best_fit else 0,
                'details': fitting_results.get(best_fit, {})
            } if best_fit else {'name': None},
            'data_points': {
                'x_data': x_data.tolist(),
                'y_data': y_data.tolist()
            }
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """🔗 相関係数の解釈"""
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "相関なし"
        elif abs_r < 0.3:
            strength = "弱い相関"
        elif abs_r < 0.7:
            strength = "中程度の相関"
        else:
            strength = "強い相関"
        
        direction = "負の" if r < 0 else "正の"
        return f"{direction}{strength}"
    
    def _create_normalization_visualization(
        self, 
        column_data: Dict, 
        function_results: Dict, 
        frame_id: str
    ) -> str:
        """🎨 正規化関数可視化の作成"""
        try:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Hiragino Sans']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f'距離正規化関数分析 📊\n{frame_id}\n{datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                fontsize=16
            )
            
            valid_columns = {k: v for k, v in column_data.items() if v['count'] > 0}
            
            if not valid_columns or 'data_points' not in function_results:
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'データ不足', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14)
                    ax.set_xticks([])
                    ax.set_yticks([])
                return ""
            
            x_data = np.array(function_results['data_points']['x_data'])
            y_data = np.array(function_results['data_points']['y_data'])
            
            # 🎯 1. メインプロット: 散布図 + 回帰線
            ax1 = axes[0, 0]
            
            # 個人データ（黒点）
            for column_num, data in valid_columns.items():
                x_positions = [column_num] * data['count']
                ax1.scatter(
                    x_positions, data['shoulder_widths'], 
                    color='black', alpha=0.6, s=30,
                    label='個人データ' if column_num == min(valid_columns.keys()) else ""
                )
            
            # 列平均（赤点）
            ax1.scatter(x_data, y_data, color='red', s=100, marker='D', 
                       label='列平均', zorder=5)
            
            # 最適回帰線
            if function_results.get('best_function', {}).get('name') and 'fitting_results' in function_results:
                best_func_name = function_results['best_function']['name']
                best_params = function_results['fitting_results'][best_func_name]['parameters']
                
                x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
                
                # 関数別の予測値計算
                if best_func_name == 'linear':
                    y_smooth = best_params[0] * x_smooth + best_params[1]
                elif best_func_name == 'exponential':
                    y_smooth = best_params[0] * np.exp(-best_params[1] * x_smooth) + best_params[2]
                elif best_func_name == 'power':
                    y_smooth = best_params[0] / (x_smooth ** best_params[1]) + best_params[2]
                elif best_func_name == 'polynomial2':
                    y_smooth = best_params[0] * x_smooth**2 + best_params[1] * x_smooth + best_params[2]
                
                ax1.plot(x_smooth, y_smooth, 'b-', linewidth=2, 
                        label=f'最適関数 ({function_results["fitting_results"][best_func_name]["name"]})')
            
            ax1.set_xlabel('列位置（カメラからの距離）')
            ax1.set_ylabel('肩幅 (pixels)')
            ax1.set_title('🎯 距離-肩幅関係と正規化関数')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 📈 2. 関数比較プロット
            ax2 = axes[0, 1]
            if 'fitting_results' in function_results:
                colors = ['blue', 'green', 'orange', 'purple']
                for i, (func_name, result) in enumerate(function_results['fitting_results'].items()):
                    if 'predicted_values' in result:
                        color = colors[i % len(colors)]
                        symbol = "↘️" if result.get('slope_direction') == 'decreasing' else "↗️"
                        ax2.plot(x_data, result['predicted_values'], 'o-', 
                                color=color, linewidth=2, markersize=6,
                                label=f"{symbol}{result['name']} (R²={result['r2_score']:.3f})")
                ax2.scatter(x_data, y_data, color='red', s=100, marker='D', 
                            label='実測値', zorder=5)
            ax2.set_xlabel('列位置')
            ax2.set_ylabel('肩幅 (pixels)')
            ax2.set_title('📊 関数フィッティング比較')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # 左下: axes[1, 0] に「なす角分布（IDごとの点＋平均）」を描画
            ax3 = axes[1, 0]
            col_nums = sorted(column_data.keys())
            col_labels = [f"列{col_num}" for col_num in col_nums]
            means = []
            for i, col_num in enumerate(col_nums):
                ids = column_data[col_num]['assigned_ids']
                sub_df = self.df[(self.df['frame'] == frame_id) & (self.df['person_id'].isin(ids))]
                if 'shoulder_head_angle' in sub_df.columns:
                    angles = sub_df['shoulder_head_angle'].dropna().values
                    # -180～+180度に収める
                    angles = ((angles + 180) % 360) - 180
                else:
                    angles = np.array([])
                # 個々のIDごとに点をプロット
                ax3.scatter([i+1]*len(angles), angles, color='black', alpha=0.7, s=30)
                # 平均値
                if len(angles) > 0:
                    mean_angle = np.mean(angles)
                    means.append(mean_angle)
                    ax3.scatter(i+1, mean_angle, color='red', s=120, marker='D', label='列平均' if i==0 else "")
                else:
                    means.append(np.nan)

            ax3.set_xticks(range(1, len(col_labels)+1))
            ax3.set_xticklabels(col_labels)
            ax3.set_title("肩の中点と頭中心のなす角分布（列位置ごと）")
            ax3.set_xlabel("列位置")
            ax3.set_ylabel("なす角 (度)")
            ax3.set_ylim(-180, 180)
            ax3.grid(True, alpha=0.3)
            if any([len(column_data[col]['assigned_ids']) > 0 for col in col_nums]):
                ax3.legend()
            
            # 🔄 4. 正規化効果プレビュー
            ax4 = axes[1, 1]
            ax4.text(0.5, 0.5, '🔄 正規化効果\n（実装中）', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('🔄 正規化効果比較')
            
            plt.tight_layout()
            
            # 💾 保存
            output_path = self.output_dir / 'distance_normalization_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 可視化保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 可視化エラー: {e}")
            return ""
    
    def plot_angle_vs_column_position(self, metrics_csv_path, column_assignments, output_dir):
        """
        列ごとにIDを指定し、なす角（shoulder_head_angle）の分布を
        横軸: 列位置（物理的な並び順）、縦軸: なす角（度）としてプロットする。
        距離-肩幅関係グラフと同じスタイルで出力。
        """
        df = pd.read_csv(metrics_csv_path)

        # 列番号順に並べる
        col_nums = sorted(column_assignments.keys())
        col_labels = [f"列{col_num}" for col_num in col_nums]
        angle_data = []
        for col_num in col_nums:
            id_list = column_assignments[col_num]
            sub_df = df[df['person_id'].isin(id_list)]
            if 'shoulder_head_angle' in sub_df.columns:
                angle_data.append(sub_df['shoulder_head_angle'].dropna().values)
            else:
                angle_data.append(np.array([]))

        # 箱ひげ図で分布を可視化
        plt.figure(figsize=(10, 6))
        plt.boxplot(angle_data, labels=col_labels, patch_artist=True,
                    boxprops=dict(facecolor='skyblue', color='navy'),
                    medianprops=dict(color='red'))
        plt.title("肩の中点と頭中心のなす角分布（列位置ごと）")
        plt.xlabel("列位置")
        plt.ylabel("なす角 (度)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "angle_vs_column_position.png"))
        plt.close()
    
    def _generate_normalization_code(self, function_results: Dict) -> str:
        """🔧 正規化関数のPythonコード生成"""
        if not function_results.get('best_function', {}).get('name'):
            return "# エラー: 有効な関数が見つかりませんでした"
        
        best_name = function_results['best_function']['name']
        best_result = function_results['fitting_results'][best_name]
        params = best_result['parameters']
        r2_score = best_result['r2_score']
        
        # ヘッダー
        code = f'''# 🔧 距離正規化関数（自動生成）
# 分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 最適関数: {best_result['name']} (R² = {r2_score:.3f})
# 想定: 列数↑ → 距離↑ → 肩幅↓ (右肩下がり)

import numpy as np

def distance_normalization_function(column_position):
    """
    📏 カメラ距離による肩幅予測関数
    
    Args:
        column_position (float): 列位置（1=前列, 2=中列, 3=後列, ...）
        
    Returns:
        float: 予測肩幅値 (pixels)
    """'''
        
        # 関数別の実装
        if best_name == 'linear':
            a, b = params[0], params[1]
            code += f'''
    # 線形関数: f(x) = {a:.3f}x + {b:.3f}
    return {a:.6f} * column_position + {b:.6f}'''
        
        elif best_name == 'exponential':
            a, b, c = params[0], params[1], params[2]
            code += f'''
    # 指数減衰関数: f(x) = {a:.3f} * exp(-{b:.3f}x) + {c:.3f}
    return {a:.6f} * np.exp(-{b:.6f} * column_position) + {c:.6f}'''
        
        elif best_name == 'power':
            a, b, c = params[0], params[1], params[2]
            code += f'''
    # べき関数: f(x) = {a:.3f} / x^{b:.3f} + {c:.3f}
    return {a:.6f} / (column_position ** {b:.6f}) + {c:.6f}'''
        
        elif best_name == 'polynomial2':
            a, b, c = params[0], params[1], params[2]
            code += f'''
    # 2次多項式: f(x) = {a:.3f}x² + {b:.3f}x + {c:.3f}
    return {a:.6f} * column_position**2 + {b:.6f} * column_position + {c:.6f}'''
        
        # 正規化関数
        code += f'''

def normalize_shoulder_width(measured_width, column_position, reference_column=1):
    """
    🎯 実測肩幅を基準列で正規化
    
    Args:
        measured_width (float): 実測肩幅値 (pixels)
        column_position (float): 測定位置の列番号
        reference_column (float): 基準列番号（デフォルト: 1=前列）
        
    Returns:
        float: 正規化された肩幅値 (1列目相当)
        
    Example:
        >>> # 3列目で測定された80pxを1列目基準で正規化
        >>> normalized = normalize_shoulder_width(80, 3, 1)
        >>> print(f"3列目80px → 1列目相当{{normalized:.1f}}px")
    """
    predicted_width = distance_normalization_function(column_position)
    reference_width = distance_normalization_function(reference_column)
    
    # 正規化倍率計算
    normalization_factor = reference_width / predicted_width
    return measured_width * normalization_factor

def get_normalization_factor(column_position, reference_column=1):
    """
    📊 正規化倍率のみを取得
    
    Returns:
        float: 正規化倍率
    """
    predicted_width = distance_normalization_function(column_position)
    reference_width = distance_normalization_function(reference_column)
    return reference_width / predicted_width

# 🧪 使用例・テスト
if __name__ == "__main__":
    print("🔧 距離正規化関数テスト")
    print("=" * 40)
    
    # 各列の予測肩幅
    for col in [1, 2, 3]:
        predicted = distance_normalization_function(col)
        print(f"{{col}}列目予測肩幅: {{predicted:.1f}}px")
    
    print("\\n📊 正規化例:")
    # 正規化例
    test_cases = [
        (80, 1),  # 1列目80px
        (70, 2),  # 2列目70px  
        (60, 3),  # 3列目60px
    ]
    
    for width, col in test_cases:
        normalized = normalize_shoulder_width(width, col, 1)
        factor = get_normalization_factor(col, 1)
        print(f"{{col}}列目{{width}}px → 1列目相当{{normalized:.1f}}px (倍率={{factor:.2f}})")
'''
        
        return code
    
    def _save_function_parameters(self, function_results: Dict) -> str:
        """💾 関数パラメータのJSON保存"""
        try:
            output_path = self.output_dir / 'function_parameters.json'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(function_results, f, indent=2, ensure_ascii=False)
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ パラメータ保存エラー: {e}")
            return ""
    
    def _save_analysis_report(
        self,
        frame_id: str,
        column_assignments: Dict,
        column_data: Dict,
        function_results: Dict,
        normalization_code: str
    ) -> str:
        """📝 詳細レポート保存"""
        try:
            report_path = self.output_dir / 'normalization_analysis_report.txt'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 🔧 距離正規化関数分析レポート\n")
                f.write("=" * 60 + "\n\n")
                
                # 基本情報
                f.write("## 📊 分析基本情報\n")
                f.write(f"- 分析日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
                f.write(f"- 対象フレーム: {frame_id}\n")
                f.write(f"- データソース: {self.csv_path}\n")
                f.write(f"- 肩幅データ列: {self.shoulder_col}\n\n")
                
                # 列構成
                f.write("## 🎯 列構成設定\n")
                for col_num, ids in column_assignments.items():
                    data = column_data.get(col_num, {})
                    f.write(f"- {col_num}列目: 割り当てID {ids}, 検出{data.get('count', 0)}人\n")
                f.write("\n")
                
                # 統計情報
                f.write("## 📋 列別詳細統計\n")
                valid_columns = {k: v for k, v in column_data.items() if v['count'] > 0}
                for col_num in sorted(valid_columns.keys()):
                    data = valid_columns[col_num]
                    f.write(f"### {col_num}列目\n")
                    f.write(f"- 平均肩幅: {data['mean_width']:.2f} pixels\n")
                    f.write(f"- 標準偏差: {data['std_width']:.2f} pixels\n")
                    f.write(f"- 範囲: {data['min_width']:.1f} - {data['max_width']:.1f} pixels\n")
                    f.write(f"- 検出人数: {data['count']}人\n")
                    f.write(f"- 個別値: {[f'{w:.1f}' for w in data['shoulder_widths']]}\n\n")
                
                # 相関分析結果
                if 'correlation' in function_results:
                    f.write("## 🔗 相関分析結果\n")
                    corr = function_results['correlation']
                    f.write(f"- 相関係数: {corr['coefficient']:.3f}\n")
                    f.write(f"- p値: {corr['p_value']:.3f}\n")
                    f.write(f"- 解釈: {corr['interpretation']}\n")
                    
                    if corr['coefficient'] < 0:
                        f.write("- ✅ 右肩下がり傾向（距離増加→肩幅減少）\n")
                    else:
                        f.write("- ⚠️ 右肩上がり傾向（想定と逆）\n")
                    f.write("\n")
                
                # 関数フィッティング結果
                if 'fitting_results' in function_results:
                    f.write("## 📈 関数フィッティング結果\n")
                    for func_name, result in function_results['fitting_results'].items():
                        direction = "↘️減少" if result.get('slope_direction') == 'decreasing' else "↗️増加"
                        f.write(f"### {result['name']} {direction}\n")
                        f.write(f"- 数式: {result['formula']}\n")
                        f.write(f"- R²スコア: {result['r2_score']:.3f}\n")
                        f.write(f"- RMSE: {result['rmse']:.2f} pixels\n")
                        f.write(f"- パラメータ: {[f'{p:.3f}' for p in result['parameters']]}\n\n")
                
                # 最適関数
                if 'best_function' in function_results and function_results['best_function'].get('name'):
                    f.write("## 🏆 選択された最適関数\n")
                    best = function_results['best_function']
                    f.write(f"- 関数名: {best.get('details', {}).get('name', best['name'])}\n")
                    f.write(f"- R²スコア: {best['r2_score']:.3f}\n")
                    if 'details' in best:
                        f.write(f"- 数式: {best['details'].get('formula', 'N/A')}\n")
                        direction = best['details'].get('slope_direction', 'unknown')
                        if direction == 'decreasing':
                            f.write("- ✅ 右肩下がり（想定通り）\n")
                        else:
                            f.write("- ⚠️ 右肩上がり（要確認）\n")
                f.write("\n")
                
                # 🎯 正規化コード
                f.write("## 🔧 生成された正規化関数コード\n")
                f.write("```python\n")
                f.write(normalization_code)
                f.write("\n```\n\n")
                
                # サマリー
                f.write("## 📋 分析サマリー\n")
                total_analyzed = sum(data['count'] for data in column_data.values())
                columns_with_data = len(valid_columns)
                
                f.write(f"- 総分析対象: {total_analyzed}人\n")
                f.write(f"- データ有効列数: {columns_with_data}列\n")
                
                if valid_columns:
                    all_widths = []
                    for data in valid_columns.values():
                        all_widths.extend(data['shoulder_widths'])
                    f.write(f"- 全体肩幅範囲: {min(all_widths):.1f} - {max(all_widths):.1f} pixels\n")
                    f.write(f"- 全体平均肩幅: {np.mean(all_widths):.2f} pixels\n")
                
                f.write(f"\n## 📁 生成ファイル一覧\n")
                f.write(f"- 可視化: distance_normalization_analysis.png\n")
                f.write(f"- 関数コード: normalization_function.py\n")
                f.write(f"- パラメータ: function_parameters.json\n")
                f.write(f"- 詳細レポート: normalization_analysis_report.txt\n")
                
                f.write(f"\n🎯 分析完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 💾 正規化関数コードを別ファイルでも保存
            code_path = self.output_dir / 'normalization_function.py'
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(normalization_code)
            
            print(f"✅ レポート保存: {report_path}")
            print(f"✅ 関数コード保存: {code_path}")
            
            return str(report_path)
            
        except Exception as e:
            print(f"❌ レポート保存エラー: {e}")
            return ""


# 🔍 ユーティリティ関数
def check_available_data(csv_path: str) -> Dict:
    """📊 利用可能なフレームとIDを確認"""
    try:
        df = pd.read_csv(csv_path)
        
        # 肩幅列の特定
        shoulder_cols = ['shoulder_width', 'shoulder_width_pixels', 'shoulder_distance']
        shoulder_col = None
        for col in shoulder_cols:
            if col in df.columns:
                shoulder_col = col
                break
        
        if shoulder_col is None:
            return {'error': f'肩幅データが見つかりません。利用可能列: {list(df.columns)}'}
        
        # フレーム別ID情報
        frame_info = {}
        for frame in df['frame'].unique():
            frame_data = df[df['frame'] == frame]
            valid_data = frame_data[
                (frame_data[shoulder_col] >= 10) & 
                (frame_data[shoulder_col] <= 500) &
                (frame_data[shoulder_col].notna())
            ]
            
            if len(valid_data) > 0:
                frame_info[frame] = {
                    'total_detections': len(frame_data),
                    'valid_shoulder_data': len(valid_data),
                    'available_ids': sorted(valid_data['person_id'].unique()),
                    'shoulder_width_range': [
                        float(valid_data[shoulder_col].min()), 
                        float(valid_data[shoulder_col].max())
                    ]
                }
        
        # 推奨フレーム（6人以上必要）
        recommended_frames = [
            frame for frame, info in frame_info.items() 
            if info['valid_shoulder_data'] >= 6
        ][:5]  # 上位5フレーム
        
        return {
            'success': True,
            'total_frames': len(frame_info),
            'shoulder_width_column': shoulder_col,
            'frame_details': frame_info,
            'recommended_frames': recommended_frames,
            'minimum_requirements': {
                'min_people_per_analysis': 6,
                'min_columns': 2,
                'recommended_people_per_column': 2
            }
        }

    except Exception as e:
        return {'error': str(e)}