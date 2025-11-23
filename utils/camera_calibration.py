def undistort_with_json(image, calib_path="configs/camera_params.json"):
    import cv2
    import numpy as np
    import json

    with open(calib_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    camera_matrix = np.array(params["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(params["distortion_coefficients"], dtype=np.float32)
    image_size = tuple(params["image_size"])
    h, w = image.shape[:2]
    # スケーリング
    if (w, h) != image_size:
        scale_x = w / image_size[0]
        scale_y = h / image_size[1]
        camera_matrix = camera_matrix.copy()
        camera_matrix[0, 0] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 2] *= scale_y
    # alpha=1.0で全ピクセル保持
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0.4, (w, h)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    # ROIで切り取らない（黒枠も含めて返す）
    return undistorted