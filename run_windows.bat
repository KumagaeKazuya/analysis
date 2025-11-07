@echo off
echo YOLO11 広角カメラ分析システム（Mediumモデル・深度推定統合版）
echo.

if not exist "yolo11_env" (
    echo 仮想環境を作成中...
    python -m venv yolo11_env
)

echo 仮想環境を有効化中...
call yolo11_env\Scripts\activate

echo 依存関係をインストール中...
pip install -r requirements.txt

echo.
echo 実行モードを選択してください:
echo 1. 基本分析（Mediumモデル・深度推定なし）
echo 2. 深度推定統合分析（Mediumモデル）
echo 3. 実験モード
echo.

set /p choice="選択肢 (1-3): "

if "%choice%"=="1" (
    echo Mediumモデルで基本分析を開始...
    python improved_main.py --mode baseline --config configs/default.yaml
) else if "%choice%"=="2" (
    echo Mediumモデルで深度推定統合分析を開始...
    python improved_main.py --mode baseline --config configs/depth_config.yaml
) else if "%choice%"=="3" (
    echo 実験タイプを選択してください:
    echo - camera_calibration
    echo - model_ensemble
    echo - data_augmentation
    echo - tile_inference_comparison
    echo - depth_analysis_comparison
    echo.
    set /p exp_type="実験タイプ: "
    python improved_main.py --mode experiment --experiment-type %exp_type%
) else (
    echo 無効な選択です。Mediumモデルで基本分析を実行します。
    python improved_main.py --mode baseline --config configs/default.yaml
)

pause
