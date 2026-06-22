#!/usr/bin/env python3
"""Refine character boxes from existing YOLO plate detections."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from license_plate_detector import fallback_even_char_boxes, segment_characters


def robust_imread(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is not None:
        return image
    try:
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def clamp_box(box: tuple[int, int, int, int], image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def expand_box(
    box: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    x_ratio: float = 0.06,
    y_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    px = int(round(w * x_ratio))
    py = int(round(h * y_ratio))
    return clamp_box((x - px, y - py, w + 2 * px, h + 2 * py), image_w, image_h)


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
            aspect_score = max(0.0, 1.0 - abs(ratio - 2.45) / 3.0)
            area_score = min(area / max(crop_area * 0.45, 1.0), 1.0)
            center_penalty = abs((x + w / 2.0) / crop_w - 0.5) * 0.25
            score = aspect_score + area_score - center_penalty
            if score > best_score:
                best_score = score
                best_box = expand_box((x, y, w, h), crop_w, crop_h)
    return best_box


def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    scale = max(1.0, min(4.0, 220 / float(h)))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(4, 4)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.convertScaleAbs(cv2.addWeighted(enhanced, 1.75, blur, -0.75, 0), alpha=1.12, beta=2)


def filter_boxes_to_face(
    boxes: list[tuple[int, int, int, int]],
    face_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    fx, fy, fw, fh = face_box
    ex, ey = fx - fw * 0.06, fy - fh * 0.10
    ew, eh = fw * 1.12, fh * 1.20
    filtered = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
            filtered.append((x, y, w, h))
    return filtered if filtered else boxes


def split_wide_boxes(
    boxes: list[tuple[int, int, int, int]],
    face_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    _, _, face_w, _ = face_box
    widths = np.array([box[2] for box in boxes], dtype=np.float32)
    heights = np.array([box[3] for box in boxes], dtype=np.float32)
    ratios = widths / np.maximum(heights, 1.0)
    normal_widths = widths[ratios <= 0.68]
    target_w = float(np.median(normal_widths)) if len(normal_widths) >= 2 else max(4.0, face_w / 9.5)

    output: list[tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        ratio = w / float(max(h, 1))
        if w <= target_w * 1.65 or ratio <= 0.62:
            output.append((x, y, w, h))
            continue
        parts = max(2, min(int(round(w / max(target_w, 1.0))), 4))
        step = w / float(parts)
        part_w = max(2, int(round(step * 0.78)))
        for idx in range(parts):
            center = x + (idx + 0.5) * step
            output.append((int(round(center - part_w / 2.0)), y, part_w, h))

    output = sorted(output, key=lambda box: box[0])
    if len(output) > 8:
        output = output[:8]
    return output


def union_box(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def normalize_char_boxes(
    boxes: list[tuple[int, int, int, int]],
    face_box: tuple[int, int, int, int],
    min_chars: int = 5,
    max_chars: int = 7,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    fx, fy, fw, fh = face_box
    boxes = sorted(boxes, key=lambda box: box[0])

    def robust_stats(current: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
        widths = np.array([box[2] for box in current], dtype=np.float32)
        heights = np.array([box[3] for box in current], dtype=np.float32)
        areas = widths * heights
        tall = heights >= max(2.0, np.median(heights) * 0.60)
        usable_widths = widths[tall & (widths >= max(2.0, fw * 0.025))]
        median_w = float(np.median(usable_widths)) if len(usable_widths) else float(np.median(widths))
        return max(median_w, fw / 14.0), float(np.median(heights)), float(np.median(areas))

    def noise_score(box: tuple[int, int, int, int], median_w: float, median_h: float, median_area: float) -> float:
        x, y, w, h = box
        area = w * h
        center_x = x + w / 2.0
        edge_distance = min(center_x - fx, fx + fw - center_x) / max(float(fw), 1.0)
        score = 0.0
        if w < max(3.0, median_w * 0.42, fw * 0.022):
            score += 2.4
        if h < max(3.0, median_h * 0.62):
            score += 1.8
        if area < max(6.0, median_area * 0.34):
            score += 2.0
        if edge_distance < 0.055 and w < median_w * 0.80:
            score += 2.2
        if x <= fx + fw * 0.025 or x + w >= fx + fw * 0.975:
            score += 0.8
        return score

    while len(boxes) > max_chars:
        median_w, median_h, median_area = robust_stats(boxes)
        scored = [
            (noise_score(box, median_w, median_h, median_area), index)
            for index, box in enumerate(boxes)
        ]
        best_noise, remove_index = max(scored, key=lambda item: item[0])
        if best_noise >= 3.2 and len(boxes) - 1 >= min_chars:
            boxes.pop(remove_index)
            continue

        best_pair: tuple[float, int] | None = None
        for index in range(len(boxes) - 1):
            left = boxes[index]
            right = boxes[index + 1]
            lx, ly, lw, lh = left
            rx, ry, rw, rh = right
            gap = rx - (lx + lw)
            merged = union_box(left, right)
            merged_w = merged[2]
            height_delta = abs(lh - rh) / max(median_h, 1.0)
            center_delta = abs((ly + lh / 2.0) - (ry + rh / 2.0)) / max(median_h, 1.0)
            narrow_bonus = 0.0
            if min(lw, rw) < median_w * 0.62:
                narrow_bonus -= 0.75
            if gap <= 1:
                narrow_bonus -= 0.35
            edge_bonus = 0.0
            if index == 0 or index == len(boxes) - 2:
                edge_bonus -= 0.25
            width_penalty = max(0.0, (merged_w - median_w * 1.85) / max(median_w, 1.0))
            gap_penalty = max(0.0, gap) / max(median_w, 1.0)
            cost = gap_penalty + width_penalty * 1.25 + height_delta * 0.35 + center_delta * 0.25 + narrow_bonus + edge_bonus
            if best_pair is None or cost < best_pair[0]:
                best_pair = (cost, index)

        if best_pair is None:
            break
        merge_index = best_pair[1]
        boxes[merge_index] = union_box(boxes[merge_index], boxes[merge_index + 1])
        boxes.pop(merge_index + 1)

    if len(boxes) > max_chars:
        median_w, median_h, median_area = robust_stats(boxes)
        boxes = sorted(
            boxes,
            key=lambda box: noise_score(box, median_w, median_h, median_area),
        )[:max_chars]
        boxes = sorted(boxes, key=lambda box: box[0])
    return boxes


def refine_chars(crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    face_box = find_plate_face_box(crop)
    fx, fy, fw, fh = face_box
    face_crop = crop[fy : fy + fh, fx : fx + fw]
    boxes, _, _, _ = segment_characters(face_crop)
    boxes = [(fx + x, fy + y, w, h) for x, y, w, h in boxes]
    boxes = filter_boxes_to_face(boxes, face_box)
    boxes = split_wide_boxes(boxes, face_box)
    boxes = normalize_char_boxes(boxes, face_box)
    if len(boxes) >= 5:
        return boxes

    boxes = fallback_even_char_boxes(face_crop, count=7)
    boxes = [(fx + x, fy + y, w, h) for x, y, w, h in boxes]
    boxes = filter_boxes_to_face(boxes, face_box)
    return normalize_char_boxes(boxes, face_box)


def write_submission_txt(rows: list[dict[str, str]], txt_path: Path) -> None:
    lines: list[str] = []
    for row in rows:
        lines.append(Path(row["image"]).stem)
        boxes = []
        raw_boxes = row.get("char_boxes", "")
        if raw_boxes:
            for item in raw_boxes.split(";"):
                if not item.strip():
                    continue
                x, y, w, h = [int(v) for v in item.split(",")]
                boxes.append((x + 1, y + 1, w, h))
        lines.append(str(len(boxes)))
        for x, y, w, h in boxes:
            lines.append(f"{x} {y} {w} {h}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = Path("outputs_yolo")
    rows_path = output_dir / "results.csv"
    with rows_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    (output_dir / "char_boxes").mkdir(exist_ok=True)
    (output_dir / "plate_processed").mkdir(exist_ok=True)

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image"])
        image = robust_imread(image_path)
        plate_box = row.get("plate_box", "")
        if image is None or not plate_box:
            row["char_count"] = "0"
            row["char_boxes"] = ""
            if not row.get("source"):
                row["source"] = "no_plate"
            continue

        x, y, w, h = [int(v) for v in plate_box.split(",")]
        x, y, w, h = clamp_box((x, y, w, h), image.shape[1], image.shape[0])
        crop = image[y : y + h, x : x + w]
        cv2.imwrite(str(output_dir / "plate_processed" / f"{image_path.stem}.jpg"), preprocess_plate_crop(crop))

        char_boxes = refine_chars(crop)
        absolute_boxes = [(x + cx, y + cy, cw, ch) for cx, cy, cw, ch in char_boxes]
        row["char_count"] = str(len(absolute_boxes))
        row["recognized"] = "?" * len(absolute_boxes)
        row["char_boxes"] = ";".join(",".join(map(str, box)) for box in absolute_boxes)
        row["source"] = "yolo_plate_refined_chars"

        vis = image.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3)
        for bx, by, bw, bh in absolute_boxes:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / "char_boxes" / f"{image_path.stem}.jpg"), vis)
        print(f"[{index}/{len(rows)}] {image_path.name}: chars={len(absolute_boxes)}", flush=True)

    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image", "plate_box", "char_count", "recognized", "char_boxes", "source", "plate_conf"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_submission_txt(rows, output_dir / "submission.txt")
    print(f"Refined outputs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
