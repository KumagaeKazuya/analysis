"""
🔧 距離正規化分析パッケージ

カメラ距離による肩幅変化を分析し、正規化関数を生成するパッケージです。

主な機能:
- 距離による肩幅減少の定量化
- 複数の数学関数による回帰分析
- 正規化関数の自動生成
- 詳細な可視化レポート生成

使用例:
    >>> from normalization_analysis import DistanceNormalizationAnalyzer
    >>> analyzer = DistanceNormalizationAnalyzer("data.csv")
    >>> result = analyzer.analyze_distance_function(
    ...     frame_id="frame1.jpg",
    ...     column_assignments={1: [1,2], 2: [3,4], 3: [5,6]}
    ... )
    
    # またはコマンドライン
    $ python -m normalization_analysis.run_normalization_analysis check data.csv
    $ python -m normalization_analysis.run_normalization_analysis analyze data.csv frame1.jpg --col1 1 2 --col2 3 4

Author: GitHub Copilot
Version: 1.0.0
Date: 2024-11-16
"""

from .distance_normalization import DistanceNormalizationAnalyzer, check_available_data

# パッケージメタデータ
__version__ = "1.0.0"
__author__ = "GitHub Copilot"
__email__ = "copilot@github.com"
__description__ = "距離正規化関数分析パッケージ"
__license__ = "MIT"

# 公開API
__all__ = [
    'DistanceNormalizationAnalyzer',
    'check_available_data',
    'run_sample_analysis',
    'get_package_info'
]

def run_sample_analysis(csv_path: str = None) -> dict:
    """
    🔬 サンプル分析の簡易実行
    
    Args:
        csv_path: CSVファイルパス（Noneの場合は自動検索）
        
    Returns:
        dict: 分析結果
        
    Example:
        >>> result = run_sample_analysis()
        >>> print(f"相関係数: {result['correlation']}")
    """
    try:
        if csv_path is None:
            # デフォルトパス検索
            from pathlib import Path
            import glob
            
            default_paths = [
                'outputs/baseline/*/4point_metrics.csv',
                'outputs/*/4point_metrics.csv',
                '4point_metrics.csv'
            ]
            
            for pattern in default_paths:
                matches = glob.glob(pattern)
                if matches:
                    csv_path = matches[0]
                    break
            
            if csv_path is None:
                return {
                    'success': False, 
                    'error': 'CSVファイルが見つかりません'
                }
        
        # データ確認
        data_info = check_available_data(csv_path)
        if 'error' in data_info:
            return {'success': False, 'error': data_info['error']}
        
        if not data_info['recommended_frames']:
            # 条件を緩めて検索
            all_frames = sorted(
                data_info['frame_details'].items(),
                key=lambda x: x[1]['valid_shoulder_data'],
                reverse=True
            )
            
            if not all_frames:
                return {
                    'success': False, 
                    'error': '分析可能なフレームがありません'
                }
            
            frame, frame_info = all_frames[0]
            available_ids = frame_info['available_ids']
        else:
            frame = data_info['recommended_frames'][0]
            available_ids = data_info['frame_details'][frame]['available_ids']
        
        if len(available_ids) < 4:
            return {
                'success': False, 
                'error': f'ID数不足: {len(available_ids)}人 (最低4人必要)'
            }
        
        # ID分割
        if len(available_ids) >= 6:
            ids_per_col = len(available_ids) // 3
            assignments = {
                1: available_ids[:ids_per_col],
                2: available_ids[ids_per_col:ids_per_col*2],
                3: available_ids[ids_per_col*2:]
            }
        else:
            ids_per_col = len(available_ids) // 2
            assignments = {
                1: available_ids[:ids_per_col],
                2: available_ids[ids_per_col:]
            }
        
        # 分析実行
        analyzer = DistanceNormalizationAnalyzer(csv_path)
        result = analyzer.analyze_distance_function(
            frame_id=frame,
            column_assignments=assignments
        )
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_package_info() -> dict:
    """
    📋 パッケージ情報を取得
    
    Returns:
        dict: パッケージ情報辞書
    """
    return {
        'name': 'normalization_analysis',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'license': __license__,
        'dependencies': [
            'pandas>=1.3.0',
            'numpy>=1.21.0', 
            'matplotlib>=3.4.0',
            'scipy>=1.7.0'
        ],
        'main_classes': [
            'DistanceNormalizationAnalyzer'
        ],
        'utility_functions': [
            'check_available_data',
            'run_sample_analysis'
        ],
        'command_line': {
            'check': 'データ確認',
            'analyze': '分析実行',
            'sample': 'サンプル分析'
        }
    }

# パッケージ読み込み時の初期化
def _initialize_package():
    """📦 パッケージ初期化処理"""
    try:
        # 必要なライブラリのチェック
        import pandas
        import numpy  
        import matplotlib
        import scipy
        
        # 正常読み込み完了
        return True
        
    except ImportError as e:
        print(f"⚠️ 依存ライブラリが不足しています: {e}")
        print("💡 以下のコマンドでインストールしてください:")
        print("pip install pandas numpy matplotlib scipy")
        return False

# 初期化実行
_package_ready = _initialize_package()

# パッケージ使用可能性フラグ
PACKAGE_READY = _package_ready

# 簡易使用ガイド
USAGE_GUIDE = """
🔧 距離正規化分析パッケージ使用ガイド

1. 📊 データ確認:
   from normalization_analysis import check_available_data
   info = check_available_data("your_data.csv")

2. 🎯 分析実行:
   from normalization_analysis import DistanceNormalizationAnalyzer
   analyzer = DistanceNormalizationAnalyzer("your_data.csv")
   result = analyzer.analyze_distance_function(
       frame_id="frame1.jpg",
       column_assignments={1: [1,2], 2: [3,4], 3: [5,6]}
   )

3. 🔬 サンプル分析:
   from normalization_analysis import run_sample_analysis
   result = run_sample_analysis()

4. 💻 コマンドライン:
   python -m normalization_analysis.run_normalization_analysis check data.csv
   python -m normalization_analysis.run_normalization_analysis analyze data.csv frame.jpg --col1 1 2 --col2 3 4

詳細: https://github.com/your-repo/normalization_analysis
"""

def print_usage_guide():
    """📖 使用ガイドを表示"""
    print(USAGE_GUIDE)

# モジュール動作確認
if __name__ == "__main__":
    print("🔧 距離正規化分析パッケージ")
    print(f"📦 バージョン: {__version__}")
    print(f"✅ パッケージ準備: {'OK' if PACKAGE_READY else 'NG'}")
    print("\n" + "="*60)
    print_usage_guide()