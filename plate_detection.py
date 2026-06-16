def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("使用 --yolo-model 需要先安裝 ultralytics：pip install ultralytics") from exc

    return YOLO(str(model_path))


def find_yolo_plate_roi(model, color_image, confidence_threshold: float, padding_ratio: float):
    results = model.predict(color_image, conf=confidence_threshold, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    image_height, image_width = color_image.shape[:2]
    best_roi = None
    best_score = -1.0

    for box in results[0].boxes:
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        area = width * height
        score = confidence * area
        if score > best_score:
            pad_x = width * padding_ratio
            pad_y = height * padding_ratio
            roi_x1 = max(0, int(round(x1 - pad_x)))
            roi_y1 = max(0, int(round(y1 - pad_y)))
            roi_x2 = min(image_width, int(round(x2 + pad_x)))
            roi_y2 = min(image_height, int(round(y2 + pad_y)))
            best_roi = (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)
            best_score = score

    return best_roi
