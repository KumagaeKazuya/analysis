#!/bin/bash
echo "YOLO11 広角カメラ分析システム（Mediumモデル・深度推定統合版）"
echo

if [ ! -d "yolo11_env" ]; then
    echo "仮想環境を作成中..."
    python3 -m venv yolo11_env
fi

echo "仮想環境を有効化中..."
source yolo11_env/bin/activate

echo "依存関係をインストール中..."
pip install -r requirements.txt

echo
echo "実行モードを選択してください:"
echo "1. 基本分析（Mediumモデル・深度推定なし）"
echo "2. 深度推定統合分析（Mediumモデル）"
echo "3. 実験モード"
echo

read -p "選択肢 (1-3): " choice

case $choice in
    1)
        echo "Mediumモデルで基本分析を開始..."
        python improved_main.py --mode baseline --config configs/default.yaml
        ;;
    2)
        echo "Mediumモデルで深度推定統合分析を開始..."
        python improved_main.py --mode baseline --config configs/depth_config.yaml
        ;;
    3)
        echo "実験タイプを選択してください:"
        echo "- camera_calibration"
        echo "- model_ensemble"
        echo "- data_augmentation"
        echo "- tile_inference_comparison"
        echo "- depth_analysis_comparison"
        echo
        read -p "実験タイプ: " exp_type
        python improved_main.py --mode experiment --experiment-type $exp_type
        ;;
    *)
        echo "無効な選択です。Mediumモデルで基本分析を実行します。"
        python improved_main.py --mode baseline --config configs/default.yaml
        ;;
esac
