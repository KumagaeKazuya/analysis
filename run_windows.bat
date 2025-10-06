@echo off
echo YOLO11 広角カメラ分析システム
echo.

if not exist "yolo11_env" (
    echo 仮想環境を作成中...
    python -m venv yolo11_env
)

echo 仮想環境を有効化中...
call yolo11_env\Scripts\activate

echo 依存関係をインストール中...
pip install -r requirements.txt

echo ベースライン分析を開始...
python improved_main.py --mode baseline --config configs/default.yaml

pause
