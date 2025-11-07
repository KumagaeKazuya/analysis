# 動画ファイルの配置について

このディレクトリ（videos/）に分析対象の動画ファイルを配置してください。

## 対応形式
- MP4 (.mp4) ✅ 推奨
- AVI (.avi)
- MOV (.mov)

## 推奨仕様
- 解像度: 1280x720以上（広角カメラを想定）
- フレームレート: 30fps以下
- ファイルサイズ: 1GB以下（メモリ効率のため）
- コーデック: H.264推奨

## 🔍 深度推定に適した動画
- 教室や会議室などの室内環境
- 固定カメラでの撮影
- 人物の全身が写っている
- 照明が安定している
- 背景に奥行きがある

## サンプル動画
広角カメラで撮影された人物が含まれる動画を配置してください。
例：
- 教室の後方から撮影した授業風景
- 会議室の全体を撮影した映像
- 店舗監視カメラの映像
- イベント会場の広角映像

## ファイル名の注意
- 日本語ファイル名は避けることを推奨
- スペースを含む場合はアンダースコア（_）に置換
- 特殊文字は避ける

## 分析コマンド（Mediumモデル仕様）

### 基本分析（深度推定なし・Mediumモデル）
```bash
python improved_main.py --mode baseline --config configs/default.yaml
```

### 🔍 深度推定統合分析（Mediumモデル）
```bash
python improved_main.py --mode baseline --config configs/depth_config.yaml
```

### 深度推定強制有効化（Mediumモデル）
```bash
python improved_main.py --mode baseline --enable-depth
```

## 実験タイプ
- `camera_calibration`: カメラキャリブレーション実験
- `model_ensemble`: モデルアンサンブル実験
- `data_augmentation`: データ拡張実験
- `tile_inference_comparison`: タイル推論比較実験
- `depth_analysis_comparison`: 深度分析比較実験（新機能）

### 実験実行例（Mediumモデル）
```bash
python improved_main.py --mode experiment --experiment-type depth_analysis_comparison
```

## 使用モデル
- 🎯 デフォルト: Medium (yolo11m.pt, yolo11m-pose.pt)
- 🚀 高性能オプション: XLarge (yolo11x.pt, yolo11x-pose.pt)
- 🔧 フォールバック: Nano (yolo11n.pt, yolo11n-pose.pt)
