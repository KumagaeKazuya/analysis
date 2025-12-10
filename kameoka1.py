import cv2
import numpy as np
import logging
import os
from datetime import datetime
import argparse

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_output_dirs(base_dir):
    video_dir = os.path.join(base_dir, "video")
    json_dir = os.path.join(base_dir, "json")
    img_dir = os.path.join(base_dir, "img")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    return video_dir, json_dir, img_dir

# 歪み補正設定（実測キャリブレーションパラメータの使用）
DISTORTION_CONFIG = {
    "use_calibrated_params": True,  # 実測キャリブレーション値の使用
    "k1": -0.30906428,
    "k2": 0.12771288,
    "p1": 0.0026938,
    "p2": 0.00175418,
    "k3": -0.03167725,
    "alpha": 0.4,  # 0.0で有効ピクセルのみ、1.0で全ピクセル保持
    "apply_correction": True,
}

class VideoDistortionCorrector:
    """動画の歪み補正クラス（実測キャリブレーション対応版）"""

    def __init__(self, use_calibrated_params=True, k1=-0.1, k2=0.0, p1=0.0, p2=0.0, k3=0.0, alpha=0.0):
        """
        歪み補正パラメータを初期化

        引数:
            use_calibrated_params: Trueの場合、実測キャリブレーション値を使用
            k1, k2, k3: 放射歪係数
            p1, p2: 接続線歪係数
            alpha: 新しいカメラマトリックのスケーリング（0.0=有効ピクセルのみ、1.0=全ピクセル）
        """
        if use_calibrated_params:
            # 実測されたキャリブレーションパラメータを使用
            self.k1 = -0.30906428
            self.k2 = 0.12771288
            self.p1 = 0.0026938
            self.p2 = 0.00175418
            self.k3 = -0.03167725

            # キャリブレーション済みカメラストリーム（1920x1080用）
            self.calibrated_camera_matrix = np.array([
                [1.14818439e+03, 0.00000000e+00, 9.17249755e+02],
                [0.00000000e+00, 1.14628046e+03, 6.18787769e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ], dtype=np.float32)

            self.use_precalibrated = True
            logger.info("✅ 実測キャリブレーションパラメーターを使用")
        else:
            # 引数で渡された推定値を使用
            self.k1 = k1
            self.k2 = k2
            self.p1 = p1
            self.p2 = p2
            self.k3 = k3
            self.calibrated_camera_matrix = None
            self.use_precalibrated = False
            logger.info("⚠️推定パラメータを使用")

        self.alpha = alpha
        self.map_x = None
        self.map_y = None
        self.camera_matrix = None
        self.new_camera_matrix = None
        self.dist_coeffs = None

        logger.info(f"歪み補正初期化:")
        logger.info(f" k1={self.k1:.6f}, k2={self.k2:.6f}, k3={self.k3:.6f}")
        logger.info(f" p1={self.p1:.6f}, p2={self.p2:.6f}")
        logger.info(f" alpha={self.alpha:.2f}")

    def create_correction_maps(self, width, height):
        """歪み補正マップを作成（キャリブレーション対応版）"""
        logger.info(f"歪み補正マップ作成開始: {width}x{height}")

        # 5の歪み係数を使用する
        self.dist_coeffs = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32
        )

        if self.use_precalibrated and self.calibrated_camera_matrix is not None:
            # 実測キャリブレーション済みカメラストリームを使用
            if width == 1920 and height == 1080:
                self.camera_matrix = self.calibrated_camera_matrix.copy()
                logger.info("✅ 実測カメラ行列を使用（1920x1080）")
            else:
                # 解像度が違う場合はスケーリング
                scale_x = width / 1920.0
                scale_y = height / 1080.0

                self.camera_matrix = self.calibrated_camera_matrix.copy()
                self.camera_matrix[0, 0] *= scale_x  # fx
                self.camera_matrix[1, 1] *= scale_y  # fy
                self.camera_matrix[0, 2] *= scale_x  # cx
                self.camera_matrix[1, 2] *= scale_y  # cy

                logger.info(f"⚠️カメラ行列をスケーリング: {scale_x:.3f}x{scale_y:.3f}")
        else:
            # 従来の推定方式
            fx = fy = min(width, height) * 0.9
            cx, cy = width / 2.0, height / 2.0

            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            logger.info("⚠️推定カメラ行列を使用")

        # 最適な新しいカメラマトリックスを計算
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (width, height),
            self.alpha,
            (width, height)
        )

        # 補正マップ作成
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            (width, height),
            cv2.CV_32FC1
        )

        logger.info("✅歪み補正マップ作成完了")
        self._log_calibration_info()
        self._log_map_statistics()

    def _log_calibration_info(self):
        """キャリブレーション情報を詳細ログ出力"""
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            logger.info("📷カメラパラメータ:")
            logger.info(f" 焦点距離: fx={fx:.2f}, fy={fy:.2f}")
            logger.info(f" 主点: cx={cx:.2f}, cy={cy:.2f}")
            logger.info(f" 歪み係数: [{self.k1:.6f}, {self.k2:.6f}, {self.p1:.6f}, {self.p2:.6f}, {self.k3:.6f}]")

    def _log_map_statistics(self):
        """マップの統計情報をログ出力"""
        if self.map_x is not None and self.map_y is not None:
            x_mean, x_std = np.mean(self.map_x), np.std(self.map_x)
            y_mean, y_std = np.mean(self.map_y), np.std(self.map_y)
            logger.info(f"補正マップ統計: X(平均={x_mean:.2f}, 標準偏差={x_std:.2f}), Y(平均={y_mean:.2f}, 標準偏差={y_std:.2f})")

    def apply_correction(self, frame):
        """フレームに歪み補正を適用"""
        if self.map_x is None or self.map_y is None:
            logger.warning("補正マップが作成されていない")
            return frame

        return cv2.remap(
            frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

def process_video(input_video_path, output_video_path, show_preview=True):
    """動画の歪み補正処理"""
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        logger.error(f"動画ファイルはありません: {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"動画情報: {width}x{height}, {fps}FPS, {total_frames}フレーム")

    distortion_corrector = None
    if DISTORTION_CONFIG["apply_correction"]:
        distortion_corrector = VideoDistortionCorrector(
            use_calibrated_params=DISTORTION_CONFIG.get("use_calibrated_params", True),
            k1=DISTORTION_CONFIG.get("k1", -0.1),
            k2=DISTORTION_CONFIG.get("k2", 0.0),
            p1=DISTORTION_CONFIG.get("p1", 0.0),
            p2=DISTORTION_CONFIG.get("p2", 0.0),
            k3=DISTORTION_CONFIG.get("k3", 0.0),
            alpha=DISTORTION_CONFIG.get("alpha", 0.0)
        )
        distortion_corrector.create_correction_maps(width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            if distortion_corrector:
                corrected_frame = distortion_corrector.apply_correction(frame)
            else:
                corrected_frame = frame

            out.write(corrected_frame)

            if show_preview:
                display_frame = np.hstack([
                    cv2.resize(frame, (480, 270)),
                    cv2.resize(corrected_frame, (480, 270))
                ])
                cv2.putText(display_frame, "オリジナル", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "修正済み", (490, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if distortion_corrector and distortion_corrector.use_precalibrated:
                    cv2.putText(display_frame, "キャリブレーション済み", (490, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow('歪み補正', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_counter % (fps * 10) == 0:
                progress = (frame_counter / total_frames) * 100
                logger.info(f"処理進捗: {progress:.1f}% ({frame_counter}/{total_frames})")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"動画処理完了: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="動画歪み補正システム")
    parser.add_argument("--input-video", type=str, required=True, help="入力動画ファイルパス")
    parser.add_argument("--show-preview", action="store_true", help="処理中にプレビュー表示")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("outputs", f"project_{timestamp}")
    video_dir, json_dir, img_dir = get_output_dirs(base_dir)
    input_basename = os.path.splitext(os.path.basename(args.input_video))[0]
    output_video_path = os.path.join(video_dir, f"output_{input_basename}.mp4")
    process_video(args.input_video, output_video_path, show_preview=args.show_preview)
    logger.info(f"保存先: {base_dir}")
    logger.info("✅動画歪み補正システム通常終了")

    # 使用動画情報をファイル保存
    info_path = os.path.join(base_dir, "video_info.txt")
    with open(info_path, "w", encoding="utf-8-sig") as f:
        f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"入力動画: {os.path.abspath(args.input_video)}\n")
        f.write(f"出力動画: {output_video_path}\n")
    logger.info(f"✅ 動画情報を {info_path} に保存しました")

if __name__ == "__main__":
    main()