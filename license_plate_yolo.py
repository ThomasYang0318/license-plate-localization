#!/usr/bin/env python3
"""Two-stage YOLO pipeline for license plate and character boxes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from license_plate_detector import (
    fallback_even_char_boxes,
    list_images,
    safe_name,
    segment_characters,
)


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    conf: float
    label: str = ""


@dataclass
class ProcessedPlate:
    image: np.ndarray
    scale_x: float
    scale_y: float


def robust_imread(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is not None:
        return image
    try:
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb = np.array(pil_image)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect plate and characters with YOLO.")
    parser.add_argument("--input", "-i", default="Datasets", help="Input image or directory.")
    parser.add_argument("--output", "-o", default="outputs_yolo", help="Output directory.")
    parser.add_argument("--plate-model", default="models/plate_yolov8.pt")
    parser.add_argument("--char-model", default="models/char_yolov8.pt")
    parser.add_argument("--plate-conf", type=float, default=0.20)
    parser.add_argument("--char-conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640, help="Plate YOLO input size.")
    parser.add_argument("--char-imgsz", type=int, default=320, help="Character YOLO input size.")
    parser.add_argument("--device", default="", help="YOLO device, e.g. cpu, mps, cuda. Empty lets Ultralytics choose.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--txt-output",
        default="",
        help="Submission txt path. Default: <output>/submission.txt",
    )
    parser.add_argument("--draw-labels", action="store_true", help="Draw predicted characters above boxes.")
    parser.add_argument(
        "--no-preprocess-plate",
        action="store_true",
        help="Run character YOLO on the raw plate crop instead of the enhanced crop.",
    )
    parser.add_argument(
        "--skip-char-yolo",
        action="store_true",
        help="Use YOLO for plate detection, then fast image-processing character boxes.",
    )
    return parser.parse_args()


def xyxy_to_xywh(values: np.ndarray, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in values]
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(x1 + 1, min(x2, image_w))
    y2 = max(y1 + 1, min(y2, image_h))
    return x1, y1, x2 - x1, y2 - y1


def expand_box(
    box: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    x_ratio: float = 0.08,
    y_ratio: float = 0.18,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    px = int(round(w * x_ratio))
    py = int(round(h * y_ratio))
    x = max(0, x - px)
    y = max(0, y - py)
    w = min(image_w - x, w + 2 * px)
    h = min(image_h - y, h + 2 * py)
    return x, y, max(1, w), max(1, h)


def clamp_box(
    box: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def expand_local_box(
    box: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    x_ratio: float = 0.04,
    y_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    px = int(round(w * x_ratio))
    py = int(round(h * y_ratio))
    return clamp_box((x - px, y - py, w + 2 * px, h + 2 * py), image_w, image_h)


def choose_plate(result, image_w: int, image_h: int) -> Detection | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best: Detection | None = None
    best_score = -1.0
    image_area = image_w * image_h
    for box in result.boxes:
        xywh = xyxy_to_xywh(box.xyxy[0].cpu().numpy(), image_w, image_h)
        x, y, w, h = xywh
        ratio = w / float(max(h, 1))
        area = w * h
        if ratio < 0.8 or ratio > 7.0:
            continue
        if area < image_area * 0.00025 or area > image_area * 0.18:
            continue
        conf = float(box.conf[0])
        ratio_score = max(0.0, 1.0 - abs(ratio - 2.45) / 3.5)
        score = conf * 2.0 + ratio_score + min(area / (image_area * 0.015), 1.0)
        if score > best_score:
            best_score = score
            best = Detection(box=xywh, conf=conf, label="license_plate")

    if best is None:
        return None
    best.box = expand_box(best.box, image_w, image_h)
    return best


def detections_from_char_result(result, crop_w: int, crop_h: int) -> list[Detection]:
    detections: list[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names
    for box in result.boxes:
        xywh = xyxy_to_xywh(box.xyxy[0].cpu().numpy(), crop_w, crop_h)
        x, y, w, h = xywh
        if h < crop_h * 0.18 or h > crop_h * 0.98:
            continue
        if w < crop_w * 0.006 or w > crop_w * 0.35:
            continue
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append(Detection(box=xywh, conf=conf, label=str(names.get(cls_id, cls_id))))

    detections = suppress_overlapping_chars(detections)
    detections = keep_main_text_row(detections, crop_h)
    if len(detections) > 8:
        detections = sorted(detections, key=lambda item: item.conf, reverse=True)[:8]
    detections = sorted(detections, key=lambda item: item.box[0])
    return detections


def suppress_overlapping_chars(detections: list[Detection]) -> list[Detection]:
    def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = float((x2 - x1) * (y2 - y1))
        smaller = float(min(aw * ah, bw * bh))
        return inter / smaller if smaller else 0.0

    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.conf, reverse=True):
        if all(overlap(det.box, other.box) < 0.35 for other in kept):
            kept.append(det)
    return sorted(kept, key=lambda item: item.box[0])


def keep_main_text_row(detections: list[Detection], crop_h: int) -> list[Detection]:
    if len(detections) <= 1:
        return detections
    centers = np.array([d.box[1] + d.box[3] / 2.0 for d in detections], dtype=np.float32)
    heights = np.array([d.box[3] for d in detections], dtype=np.float32)
    median_center = float(np.median(centers))
    median_h = float(np.median(heights))
    tolerance = max(crop_h * 0.18, median_h * 0.65)
    filtered = [
        det
        for det, center in zip(detections, centers)
        if abs(float(center) - median_center) <= tolerance
    ]
    return filtered if filtered else detections


def find_plate_face_box(crop: np.ndarray) -> tuple[int, int, int, int]:
    crop_h, crop_w = crop.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return (0, 0, 1, 1)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (35, 25, 5), (105, 255, 235)),
        cv2.inRange(hsv, (0, 0, 75), (180, 175, 255)),
    ]

    best_box = (0, 0, crop_w, crop_h)
    best_score = -1.0
    crop_area = float(crop_w * crop_h)
    for mask in masks:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
            iterations=2,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = float(w * h)
            ratio = w / float(h)
            if ratio < 0.85 or ratio > 6.0:
                continue
            if area < crop_area * 0.08 or area > crop_area * 0.98:
                continue
            center_penalty = abs((x + w / 2.0) / crop_w - 0.5) * 0.25
            aspect_score = max(0.0, 1.0 - abs(ratio - 2.45) / 3.0)
            area_score = min(area / max(crop_area * 0.45, 1.0), 1.0)
            score = aspect_score + area_score - center_penalty
            if score > best_score:
                best_score = score
                best_box = expand_local_box((x, y, w, h), crop_w, crop_h, x_ratio=0.06, y_ratio=0.10)

    return best_box


def filter_detections_to_face(
    detections: list[Detection],
    face_box: tuple[int, int, int, int],
) -> list[Detection]:
    if not detections:
        return []
    fx, fy, fw, fh = face_box
    expanded = (fx - fw * 0.06, fy - fh * 0.10, fw * 1.12, fh * 1.20)
    ex, ey, ew, eh = expanded
    filtered = []
    for det in detections:
        x, y, w, h = det.box
        cx = x + w / 2.0
        cy = y + h / 2.0
        if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
            filtered.append(det)
    return filtered if filtered else detections


def split_wide_detections(
    detections: list[Detection],
    face_box: tuple[int, int, int, int],
) -> list[Detection]:
    if not detections:
        return []
    _, _, face_w, _ = face_box
    widths = np.array([d.box[2] for d in detections], dtype=np.float32)
    heights = np.array([d.box[3] for d in detections], dtype=np.float32)
    ratios = widths / np.maximum(heights, 1.0)
    normal_widths = widths[ratios <= 0.68]
    if len(normal_widths) >= 2:
        target_w = float(np.median(normal_widths))
    else:
        target_w = max(4.0, face_w / 9.5)

    output: list[Detection] = []
    for det in detections:
        x, y, w, h = det.box
        ratio = w / float(max(h, 1))
        should_split = w > target_w * 1.65 and ratio > 0.62
        if not should_split:
            output.append(det)
            continue

        parts = int(round(w / max(target_w, 1.0)))
        parts = max(2, min(parts, 4))
        step = w / float(parts)
        part_w = max(2, int(round(step * 0.78)))
        for idx in range(parts):
            center = x + (idx + 0.5) * step
            px = int(round(center - part_w / 2.0))
            output.append(Detection(box=(px, y, part_w, h), conf=det.conf * 0.9, label="?"))

    output = sorted(output, key=lambda item: item.box[0])
    if len(output) > 8:
        output = sorted(output, key=lambda item: item.conf, reverse=True)[:8]
        output = sorted(output, key=lambda item: item.box[0])
    return output


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def preprocess_plate_crop(crop: np.ndarray, target_height: int = 220) -> ProcessedPlate:
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return ProcessedPlate(crop.copy(), 1.0, 1.0)

    scale = max(1.0, min(4.0, target_height / float(h)))
    if scale > 1.0:
        work = cv2.resize(
            crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        work = crop.copy()

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))
    if v_mean < 85:
        work = apply_gamma(work, 0.68)
    elif v_mean > 185:
        work = apply_gamma(work, 1.18)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(4, 4))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharp = cv2.addWeighted(denoised, 1.75, blur, -0.75, 0)
    sharp = cv2.convertScaleAbs(sharp, alpha=1.12, beta=2)

    return ProcessedPlate(sharp, sharp.shape[1] / float(w), sharp.shape[0] / float(h))


def remap_detections_to_crop(
    detections: list[Detection],
    processed: ProcessedPlate,
    crop_w: int,
    crop_h: int,
) -> list[Detection]:
    remapped: list[Detection] = []
    for det in detections:
        x, y, w, h = det.box
        mapped = (
            int(round(x / processed.scale_x)),
            int(round(y / processed.scale_y)),
            int(round(w / processed.scale_x)),
            int(round(h / processed.scale_y)),
        )
        mapped = (
            max(0, min(mapped[0], crop_w - 1)),
            max(0, min(mapped[1], crop_h - 1)),
            max(1, min(mapped[2], crop_w - max(0, min(mapped[0], crop_w - 1)))),
            max(1, min(mapped[3], crop_h - max(0, min(mapped[1], crop_h - 1)))),
        )
        remapped.append(Detection(box=mapped, conf=det.conf, label=det.label))
    return remapped


def fallback_chars(
    plate_crop: np.ndarray,
    face_box: tuple[int, int, int, int] | None = None,
) -> list[Detection]:
    if face_box is None:
        face_box = find_plate_face_box(plate_crop)
    fx, fy, fw, fh = face_box
    face_crop = plate_crop[fy : fy + fh, fx : fx + fw]
    boxes, _, _, _ = segment_characters(face_crop)
    detections = [
        Detection(box=(fx + x, fy + y, w, h), conf=0.0, label="?")
        for x, y, w, h in boxes
    ]
    detections = filter_detections_to_face(detections, face_box)
    detections = split_wide_detections(detections, face_box)

    if len(detections) >= 5:
        return detections

    boxes = fallback_even_char_boxes(face_crop, count=7)
    detections = [
        Detection(box=(fx + x, fy + y, w, h), conf=0.0, label="?")
        for x, y, w, h in boxes
    ]
    return filter_detections_to_face(detections, face_box)


def draw_outputs(
    image: np.ndarray,
    plate: Detection | None,
    chars: list[Detection],
    output_dir: Path,
    stem: str,
    draw_labels: bool = False,
) -> tuple[str, str, int, str]:
    h, w = image.shape[:2]
    stage1_mask = np.zeros((h, w), dtype=np.uint8)
    foreground = np.zeros_like(image)
    plate_vis = image.copy()
    char_vis = image.copy()

    plate_box_text = ""
    char_box_parts: list[str] = []
    recognized = ""

    if plate is not None:
        x, y, bw, bh = plate.box
        cv2.rectangle(stage1_mask, (x, y), (x + bw, y + bh), 255, thickness=-1)
        foreground[stage1_mask > 0] = image[stage1_mask > 0]
        cv2.rectangle(plate_vis, (x, y), (x + bw, y + bh), (0, 255, 255), 3)
        cv2.rectangle(char_vis, (x, y), (x + bw, y + bh), (0, 255, 255), 3)
        plate_box_text = ",".join(map(str, plate.box))

        for char in chars:
            cx, cy, cw, ch = char.box
            absolute = (x + cx, y + cy, cw, ch)
            char_box_parts.append(",".join(map(str, absolute)))
            recognized += char.label
            cv2.rectangle(
                char_vis,
                (absolute[0], absolute[1]),
                (absolute[0] + absolute[2], absolute[1] + absolute[3]),
                (0, 255, 0),
                2,
            )
            if draw_labels and char.label and char.label != "?":
                cv2.putText(
                    char_vis,
                    char.label,
                    (absolute[0], max(12, absolute[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    cv2.imwrite(str(output_dir / "stage1_mask" / f"{stem}.png"), stage1_mask)
    cv2.imwrite(str(output_dir / "stage1_foreground" / f"{stem}.jpg"), foreground)
    cv2.imwrite(str(output_dir / "plate_box" / f"{stem}.jpg"), plate_vis)
    cv2.imwrite(str(output_dir / "char_boxes" / f"{stem}.jpg"), char_vis)

    return plate_box_text, ";".join(char_box_parts), len(chars), recognized


def write_submission_txt(rows: list[dict[str, object]], txt_path: Path) -> None:
    lines: list[str] = []
    for row in rows:
        stem = Path(str(row["image"])).stem
        raw_boxes = str(row.get("char_boxes", ""))
        boxes: list[tuple[int, int, int, int]] = []
        if raw_boxes:
            for item in raw_boxes.split(";"):
                if not item.strip():
                    continue
                values = [int(v) for v in item.split(",")]
                if len(values) != 4:
                    continue
                x, y, w, h = values
                boxes.append((x + 1, y + 1, w, h))

        lines.append(stem)
        lines.append(str(len(boxes)))
        for x, y, w, h in boxes:
            lines.append(f"{x} {y} {w} {h}")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_image(
    image_path: Path,
    plate_model: YOLO,
    char_model: YOLO | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    image = robust_imread(image_path)
    if image is None:
        raise ValueError(f"Could not read {image_path}")

    h, w = image.shape[:2]
    predict_kwargs = {"verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device
    plate_result = plate_model.predict(
        image, imgsz=args.imgsz, conf=args.plate_conf, **predict_kwargs
    )[0]
    plate = choose_plate(plate_result, w, h)
    chars: list[Detection] = []
    source = "yolo"

    if plate is not None:
        x, y, bw, bh = plate.box
        crop = image[y : y + bh, x : x + bw]
        cv2.imwrite(str(Path(args.output) / "plate_crop" / f"{safe_name(image_path)}.jpg"), crop)
        face_box = find_plate_face_box(crop)
        processed = preprocess_plate_crop(crop)
        cv2.imwrite(str(Path(args.output) / "plate_processed" / f"{safe_name(image_path)}.jpg"), processed.image)
        if char_model is None:
            source = "yolo_plate_fallback_chars"
            chars = fallback_chars(crop, face_box)
        else:
            char_input = crop if args.no_preprocess_plate else processed.image
            char_result = char_model.predict(
                char_input, imgsz=args.char_imgsz, conf=args.char_conf, **predict_kwargs
            )[0]
            chars = detections_from_char_result(char_result, char_input.shape[1], char_input.shape[0])
            if not args.no_preprocess_plate:
                chars = remap_detections_to_crop(chars, processed, crop.shape[1], crop.shape[0])
            chars = filter_detections_to_face(chars, face_box)
            chars = split_wide_detections(chars, face_box)
            if len(chars) < 5:
                source = "yolo_plate_fallback_chars"
                chars = fallback_chars(crop, face_box)
    else:
        source = "no_plate"

    plate_box, char_boxes, char_count, recognized = draw_outputs(
        image, plate, chars, Path(args.output), safe_name(image_path), args.draw_labels
    )
    return {
        "image": str(image_path),
        "plate_box": plate_box,
        "char_count": char_count,
        "recognized": recognized,
        "char_boxes": char_boxes,
        "source": source,
        "plate_conf": f"{plate.conf:.4f}" if plate is not None else "",
    }


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "stage1_mask",
        "stage1_foreground",
        "plate_box",
        "char_boxes",
        "plate_crop",
        "plate_processed",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    images = list_images(Path(args.input))
    if args.limit > 0:
        images = images[: args.limit]

    plate_model = YOLO(args.plate_model)
    char_model = None if args.skip_char_yolo else YOLO(args.char_model)

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        try:
            row = process_image(image_path, plate_model, char_model, args)
        except Exception as exc:
            row = {
                "image": str(image_path),
                "plate_box": "",
                "char_count": 0,
                "recognized": "",
                "char_boxes": "",
                "source": f"error: {exc}",
                "plate_conf": "",
            }
        rows.append(row)
        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"chars={row['char_count']} text={row['recognized']} source={row['source']}",
            flush=True,
        )

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image", "plate_box", "char_count", "recognized", "char_boxes", "source", "plate_conf"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = Path(args.txt_output) if args.txt_output else output_dir / "submission.txt"
    write_submission_txt(rows, txt_path)
    print(f"Results written to {output_dir.resolve()}")
    print(f"Submission txt written to {txt_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
