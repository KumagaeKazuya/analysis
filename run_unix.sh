#!/bin/bash
echo "YOLO11 広角カメラ分析システム"
echo

if [ ! -d "yolo11_env" ]; then
    echo "仮想環境を作成中..."
    python3 -m venv yolo11_env
fi

echo "仮想環境を有効化中..."
source yolo11_env/bin/activate

echo "依存関係をインストール中..."
pip install -r requirements.txt

echo "ベースライン分析を開始..."
python improved_main.py --mode baseline --config configs/default.yaml
