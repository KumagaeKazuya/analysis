import cv2
import numpy as np

def draw_detections(frame, results, online_targets=None):
    """旧版のBYTETracker用の描画関数（後方互換性のため保持）"""
    # バウンディングボックス描画
    for r in results:
        if r.boxes is not None:
            for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ID表示
    if online_targets:
        for t in online_targets:
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return frame

# 🆕 新規追加: 統計グラフ生成関数
import matplotlib.pyplot as plt
import seaborn as sns

def create_detection_statistics_plot(df, output_path):
    """検出統計のグラフを生成"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 信頼度分布
    axes[0, 0].hist(df['conf'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('信頼度分布')
    axes[0, 0].set_xlabel('信頼度')
    axes[0, 0].set_ylabel('頻度')

    # ID別検出数
    id_counts = df['person_id'].value_counts().head(10)
    axes[0, 1].bar(range(len(id_counts)), id_counts.values, color='green')
    axes[0, 1].set_title('ID別検出数 (上位10)')
    axes[0, 1].set_xlabel('Person ID')
    axes[0, 1].set_ylabel('検出数')

    # バウンディングボックスサイズ分布
    df['box_area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
    axes[1, 0].hist(df['box_area'], bins=50, alpha=0.7, color='red')
    axes[1, 0].set_title('検出ボックスサイズ分布')
    axes[1, 0].set_xlabel('面積')
    axes[1, 0].set_ylabel('頻度')

    # フレーム別検出数
    frame_counts = df.groupby('frame').size()
    axes[1, 1].plot(frame_counts.values, color='purple')
    axes[1, 1].set_title('フレーム別検出数')
    axes[1, 1].set_xlabel('フレーム番号')
    axes[1, 1].set_ylabel('検出数')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_tracking_analysis_plot(df, output_path):
    """追跡分析のグラフを生成"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ID継続時間分析
    id_durations = df.groupby('person_id')['frame'].nunique()
    axes[0, 0].hist(id_durations, bins=30, alpha=0.7, color='orange')
    axes[0, 0].set_title('ID継続時間分布')
    axes[0, 0].set_xlabel('継続フレーム数')
    axes[0, 0].set_ylabel('ID数')

    # 時系列での総検出数
    temporal_counts = df.groupby('frame').size().reset_index()
    temporal_counts['frame_num'] = range(len(temporal_counts))
    axes[0, 1].plot(temporal_counts['frame_num'], temporal_counts[0], color='blue')
    axes[0, 1].set_title('時系列検出数変化')
    axes[0, 1].set_xlabel('時間')
    axes[0, 1].set_ylabel('検出数')

    # 信頼度の時系列変化
    temporal_conf = df.groupby('frame')['conf'].mean().reset_index()
    temporal_conf['frame_num'] = range(len(temporal_conf))
    axes[1, 0].plot(temporal_conf['frame_num'], temporal_conf['conf'], color='green')
    axes[1, 0].set_title('平均信頼度の時系列変化')
    axes[1, 0].set_xlabel('時間')
    axes[1, 0].set_ylabel('平均信頼度')

    # ID数の時系列変化
    temporal_ids = df.groupby('frame')['person_id'].nunique().reset_index()
    temporal_ids['frame_num'] = range(len(temporal_ids))
    axes[1, 1].plot(temporal_ids['frame_num'], temporal_ids['person_id'], color='red')
    axes[1, 1].set_title('同時追跡ID数の変化')
    axes[1, 1].set_xlabel('時間')
    axes[1, 1].set_ylabel('同時ID数')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_detections_ultralytics(frame, results):
    """Ultralytics組み込みトラッカー用の描画関数"""
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            # トラックIDがある場合
            if r.boxes.id is not None:
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for box, conf, track_id in zip(boxes, confidences, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # バウンディングボックス描画
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 信頼度表示
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # トラックID表示
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1-25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # トラックIDがない場合（通常の検出結果）
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # キーポイントがある場合の描画
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            keypoints = r.keypoints.xy.cpu().numpy()
            for kpts in keypoints:
                for x, y in kpts:
                    if x > 0 and y > 0:  # 有効なキーポイントのみ
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    return frame