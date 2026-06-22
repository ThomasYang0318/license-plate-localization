#!/usr/bin/env python3
"""License plate localization and character box extraction.

Outputs four useful debug artifacts per image:
  1. stage1_mask: binary mask where the detected plate is foreground.
  2. stage1_foreground: source image with only the detected plate visible.
  3. plate_box: source image with the detected plate rectangle.
  4. char_boxes: source image with detected character rectangles.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATE_ASPECT_TARGET = 2.45
PLATE_ASPECT_MIN = 1.05
PLATE_ASPECT_MAX = 4.50


@dataclass
class PlateCandidate:
    box: tuple[int, int, int, int]
    score: float
    source: str
    char_boxes: list[tuple[int, int, int, int]]
    char_mask: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate license plates and draw character boxes using OpenCV."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="Datasets",
        help="Input image file or directory. Default: Datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs",
        help="Directory for visual outputs and results.csv. Default: outputs",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1000,
        help="Resize long images to this width for faster processing. Default: 1000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images. 0 means all images.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also write intermediate candidate masks.",
    )
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_keep_aspect(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if max_width <= 0 or w <= max_width:
        return image.copy(), 1.0
    scale = max_width / float(w)
    resized = cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def odd_at_least(value: int, minimum: int) -> int:
    value = max(value, minimum)
    return value if value % 2 else value + 1


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def expand_box(
    box: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    x_ratio: float = 0.06,
    y_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pad_x = int(round(w * x_ratio))
    pad_y = int(round(h * y_ratio))
    return clamp_box((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), image_w, image_h)


def rescale_box(
    box: tuple[int, int, int, int], inv_scale: float
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (
        int(round(x * inv_scale)),
        int(round(y * inv_scale)),
        int(round(w * inv_scale)),
        int(round(h * inv_scale)),
    )


def nms_boxes(
    candidates: list[tuple[tuple[int, int, int, int], float, str]],
    iou_threshold: float = 0.45,
) -> list[tuple[tuple[int, int, int, int], float, str]]:
    def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = float((x2 - x1) * (y2 - y1))
        union = float(aw * ah + bw * bh - inter)
        return inter / union if union else 0.0

    kept: list[tuple[tuple[int, int, int, int], float, str]] = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(iou(candidate[0], other[0]) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def plate_candidate_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)

    rect_w = odd_at_least(w // 45, 17)
    rect_h = odd_at_least(h // 160, 5)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_w, rect_h))

    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rect_kernel)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, rect_kernel)
    strokes = cv2.max(blackhat, tophat)

    grad_x = cv2.Sobel(strokes, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.absolute(grad_x)
    min_val, max_val = float(np.min(grad_x)), float(np.max(grad_x))
    if max_val > min_val:
        grad_x = ((grad_x - min_val) / (max_val - min_val) * 255).astype("uint8")
    else:
        grad_x = np.zeros_like(gray)

    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    _, sobel_mask = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (odd_at_least(w // 38, 21), odd_at_least(h // 180, 5))
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    edge_mask = cv2.dilate(edge_mask, open_kernel, iterations=1)

    dark_plate = cv2.morphologyEx(
        blur,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (odd_at_least(w // 58, 17), odd_at_least(h // 95, 7))
        ),
    )
    _, dark_plate = cv2.threshold(dark_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark_plate = cv2.morphologyEx(
        dark_plate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (odd_at_least(w // 40, 23), odd_at_least(h // 150, 5))
        ),
        iterations=2,
    )
    dark_plate = cv2.morphologyEx(dark_plate, cv2.MORPH_OPEN, open_kernel, iterations=1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 25, 5), (105, 255, 230))
    white = cv2.inRange(hsv, (0, 0, 135), (180, 95, 255))

    color_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (odd_at_least(w // 70, 13), odd_at_least(h // 180, 5))
    )
    color_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, color_close, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, color_open, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, color_close, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, color_open, iterations=1)

    combined = cv2.bitwise_or(edge_mask, green)
    combined = cv2.bitwise_or(combined, white)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return {
        "edge": edge_mask,
        "dark_text": dark_plate,
        "green": green,
        "white": white,
        "combined": combined,
    }


def find_raw_plate_boxes(
    image: np.ndarray, masks: dict[str, np.ndarray]
) -> list[tuple[tuple[int, int, int, int], float, str]]:
    img_h, img_w = image.shape[:2]
    image_area = float(img_w * img_h)
    candidates: list[tuple[tuple[int, int, int, int], float, str]] = []

    for source, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = float(w * h)
            ratio = w / float(h)
            if ratio < PLATE_ASPECT_MIN or ratio > PLATE_ASPECT_MAX:
                continue
            if area < image_area * 0.00045 or area > image_area * 0.10:
                continue
            if w < img_w * 0.045 or h < img_h * 0.018:
                continue

            contour_area = max(cv2.contourArea(contour), 1.0)
            extent = contour_area / area
            if source in {"green", "white"} and extent < 0.22:
                continue
            if source in {"edge", "combined"} and extent < 0.08:
                continue

            aspect_score = max(0.0, 1.0 - abs(ratio - PLATE_ASPECT_TARGET) / PLATE_ASPECT_TARGET)
            area_score = min(area / (image_area * 0.018), 1.0)
            y_center = (y + h / 2.0) / img_h
            lower_half_score = 0.35 + 0.65 * y_center
            base_score = 0.8 * aspect_score + 0.45 * area_score + 0.25 * lower_half_score
            if y_center < 0.18:
                base_score -= 1.1
            if source == "green":
                base_score += 0.35
            elif source == "edge":
                base_score += 0.2

            if source == "white":
                box = expand_box((x, y, w, h), img_w, img_h, x_ratio=0.38, y_ratio=0.22)
            elif source == "green":
                box = expand_box((x, y, w, h), img_w, img_h, x_ratio=0.10, y_ratio=0.10)
            else:
                box = expand_box((x, y, w, h), img_w, img_h, x_ratio=0.10, y_ratio=0.14)
            candidates.append((box, base_score, source))

    return nms_boxes(candidates)


def clean_char_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.copy()
    h, w = cleaned.shape[:2]
    border = max(2, int(round(min(h, w) * 0.015)))
    cleaned[:border, :] = 0
    cleaned[-border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, -border:] = 0
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    return cleaned


def merge_close_boxes(
    boxes: list[tuple[int, int, int, int]], crop_w: int
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes)
    merged: list[tuple[int, int, int, int]] = [boxes[0]]
    for box in boxes[1:]:
        x, y, w, h = box
        px, py, pw, ph = merged[-1]
        gap = x - (px + pw)
        overlap_y = min(y + h, py + ph) - max(y, py)
        min_h = max(1, min(h, ph))
        should_merge = gap <= max(2, int(crop_w * 0.008)) and overlap_y > min_h * 0.35
        if should_merge:
            x1 = min(px, x)
            y1 = min(py, y)
            x2 = max(px + pw, x + w)
            y2 = max(py + ph, y + h)
            merged[-1] = (x1, y1, x2 - x1, y2 - y1)
        else:
            merged.append(box)
    return merged


def filter_boxes_to_main_row(
    boxes: list[tuple[int, int, int, int]], crop_h: int
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes
    centers = np.array([y + h / 2.0 for _, y, _, h in boxes], dtype=np.float32)
    heights = np.array([h for _, _, _, h in boxes], dtype=np.float32)
    median_center = float(np.median(centers))
    median_height = float(np.median(heights))
    tolerance = max(crop_h * 0.18, median_height * 0.38)
    filtered = [
        box
        for box, center in zip(boxes, centers)
        if abs(float(center) - median_center) <= tolerance
    ]
    return filtered if filtered else boxes


def dedupe_char_boxes(
    boxes: list[tuple[int, int, int, int]], overlap_threshold: float = 0.28
) -> list[tuple[int, int, int, int]]:
    def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = float((x2 - x1) * (y2 - y1))
        smaller = float(min(aw * ah, bw * bh))
        return inter / smaller if smaller else 0.0

    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(overlap(box, kept_box) < overlap_threshold for kept_box in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: b[0])


def extract_char_boxes_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < h * 0.28 or bh > h * 0.92:
            continue
        if bw < w * 0.012 or bw > w * 0.28:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio < 0.08 or ratio > 1.1:
            continue
        area = cv2.contourArea(contour)
        if area < w * h * 0.0012:
            continue
        if y <= 1 or y + bh >= h - 1:
            continue
        boxes.append((x, y, bw, bh))

    boxes = merge_close_boxes(boxes, w)
    boxes = filter_boxes_to_main_row(boxes, h)
    boxes = sorted(boxes, key=lambda b: b[0])

    if len(boxes) > 8:
        median_h = float(np.median([b[3] for b in boxes]))
        boxes = [
            b
            for b in boxes
            if b[3] >= median_h * 0.72 and b[2] * b[3] >= w * h * 0.002
        ]
        boxes = sorted(boxes, key=lambda b: b[0])

    return boxes


def extract_char_boxes_mser(crop: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    min_area = max(12, int(w * h * 0.0010))
    max_area = max(min_area + 1, int(w * h * 0.11))
    mser = cv2.MSER_create(5, min_area, max_area)

    boxes: list[tuple[int, int, int, int]] = []
    for image in (gray, 255 - gray):
        _, rects = mser.detectRegions(image)
        for x, y, bw, bh in rects:
            x, y, bw, bh = int(x), int(y), int(bw), int(bh)
            if bh < h * 0.22 or bh > h * 0.90:
                continue
            if bw < w * 0.010 or bw > w * 0.24:
                continue
            ratio = bw / float(max(bh, 1))
            if ratio < 0.07 or ratio > 1.05:
                continue
            if y <= 1 or y + bh >= h - 1:
                continue
            boxes.append((x, y, bw, bh))

    boxes = dedupe_char_boxes(boxes)
    boxes = filter_boxes_to_main_row(boxes, h)
    if len(boxes) > 8:
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:8]
        boxes = sorted(boxes, key=lambda b: b[0])

    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in boxes:
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, thickness=-1)
    return boxes, mask


def score_char_boxes(boxes: list[tuple[int, int, int, int]], crop_shape: tuple[int, int]) -> float:
    h, w = crop_shape
    count = len(boxes)
    if count == 0:
        return -5.0

    if 5 <= count <= 7:
        count_score = 6.0
    elif count in {4, 8}:
        count_score = 3.2
    elif count in {3, 9}:
        count_score = 1.2
    else:
        count_score = -1.5 - abs(count - 6) * 0.4

    heights = np.array([b[3] for b in boxes], dtype=np.float32)
    centers = np.array([b[1] + b[3] / 2.0 for b in boxes], dtype=np.float32)
    width_sum = sum(b[2] for b in boxes)
    span = (boxes[-1][0] + boxes[-1][2] - boxes[0][0]) / float(max(w, 1))
    height_consistency = 1.0 - min(float(np.std(heights) / max(np.mean(heights), 1.0)), 1.0)
    row_consistency = 1.0 - min(float(np.std(centers) / max(h, 1)), 1.0)
    fill_score = min(width_sum / float(max(w, 1)) * 2.6, 1.4)

    return count_score + span * 1.6 + height_consistency * 1.1 + row_consistency + fill_score


def candidate_char_masks(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    _, otsu_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, otsu_light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive_dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    adaptive_light = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5
    )

    white_text = cv2.inRange(hsv, (0, 0, 75), (180, 210, 255))
    green_bg = cv2.inRange(hsv, (35, 25, 5), (105, 255, 230))
    white_on_green = cv2.bitwise_and(white_text, cv2.bitwise_not(green_bg))

    dark_by_value = cv2.inRange(v_ch, 0, max(75, int(np.percentile(v_ch, 40))))
    dark_low_sat_guard = cv2.bitwise_or(dark_by_value, cv2.inRange(s_ch, 0, 115))
    dark_text = cv2.bitwise_and(otsu_dark, dark_low_sat_guard)

    return [
        ("dark_otsu", otsu_dark),
        ("light_otsu", otsu_light),
        ("dark_adaptive", adaptive_dark),
        ("light_adaptive", adaptive_light),
        ("white_on_green", white_on_green),
        ("dark_text", dark_text),
    ]


def segment_characters(
    plate_crop: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, str, float]:
    if plate_crop.size == 0:
        return [], np.zeros((1, 1), dtype=np.uint8), "none", -5.0

    h, w = plate_crop.shape[:2]
    scale = 1.0
    work = plate_crop
    target_h = 120
    if h > 0 and abs(h - target_h) > 10:
        scale = target_h / float(h)
        work = cv2.resize(plate_crop, (max(1, int(round(w * scale))), target_h), interpolation=cv2.INTER_CUBIC)

    best_name = "none"
    best_mask = np.zeros(work.shape[:2], dtype=np.uint8)
    best_boxes: list[tuple[int, int, int, int]] = []
    best_score = -999.0

    for name, raw_mask in candidate_char_masks(work):
        mask = clean_char_mask(raw_mask)
        boxes = extract_char_boxes_from_mask(mask)
        score = score_char_boxes(boxes, mask.shape[:2])
        if score > best_score:
            best_name = name
            best_mask = mask
            best_boxes = boxes
            best_score = score

    mser_boxes, mser_mask = extract_char_boxes_mser(work)
    mser_score = score_char_boxes(mser_boxes, work.shape[:2])
    if mser_score > best_score:
        best_name = "mser"
        best_mask = mser_mask
        best_boxes = mser_boxes
        best_score = mser_score

    inv_scale = 1.0 / scale
    original_boxes = [rescale_box(box, inv_scale) for box in best_boxes]
    if scale != 1.0:
        best_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return original_boxes, best_mask, best_name, best_score


def estimate_plate_color_score(crop: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 25, 5), (105, 255, 230))
    white = cv2.inRange(hsv, (0, 0, 135), (180, 95, 255))
    total = float(crop.shape[0] * crop.shape[1])
    return float(np.count_nonzero(green)) / total, float(np.count_nonzero(white)) / total


def refine_green_plate_box(
    image: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    img_h, img_w = image.shape[:2]
    x, y, w, h = clamp_box(box, img_w, img_h)
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return box

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 25, 5), (105, 255, 230))
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float(crop.shape[0] * crop.shape[1])
    best: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw <= 0 or rh <= 0:
            continue
        area = float(rw * rh)
        ratio = rw / float(rh)
        if ratio < 1.05 or ratio > PLATE_ASPECT_MAX:
            continue
        if area < crop_area * 0.045:
            continue
        aspect_score = max(0.0, 1.0 - abs(ratio - PLATE_ASPECT_TARGET) / PLATE_ASPECT_TARGET)
        area_score = min(area / max(crop_area * 0.25, 1.0), 1.0)
        score = aspect_score + area_score
        if score > best_score:
            best = (rx, ry, rw, rh)
            best_score = score

    if best is None:
        return box

    rx, ry, rw, rh = expand_box(best, crop.shape[1], crop.shape[0], x_ratio=0.04, y_ratio=0.04)
    return clamp_box((x + rx, y + ry, rw, rh), img_w, img_h)


def refine_light_plate_box(
    image: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    img_h, img_w = image.shape[:2]
    x, y, w, h = clamp_box(box, img_w, img_h)
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return box

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    light = cv2.inRange(hsv, (0, 0, 70), (180, 150, 255))
    light = cv2.morphologyEx(
        light,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float(crop.shape[0] * crop.shape[1])
    best: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw <= 0 or rh <= 0:
            continue
        area = float(rw * rh)
        ratio = rw / float(rh)
        if ratio < 0.70 or ratio > PLATE_ASPECT_MAX:
            continue
        if area < crop_area * 0.035 or area > crop_area * 0.88:
            continue
        aspect_score = max(0.0, 1.0 - abs(ratio - PLATE_ASPECT_TARGET) / PLATE_ASPECT_TARGET)
        compact_score = min(area / max(crop_area * 0.35, 1.0), 1.0)
        score = aspect_score + compact_score
        if score > best_score:
            best = (rx, ry, rw, rh)
            best_score = score

    if best is None:
        return box

    rx, ry, rw, rh = expand_box(best, crop.shape[1], crop.shape[0], x_ratio=0.35, y_ratio=0.10)
    return clamp_box((x + rx, y + ry, rw, rh), img_w, img_h)


def choose_plate_candidate(image: np.ndarray, debug_masks: dict[str, np.ndarray]) -> PlateCandidate | None:
    raw_boxes = find_raw_plate_boxes(image, debug_masks)
    if not raw_boxes:
        return None

    img_h, img_w = image.shape[:2]
    scored: list[PlateCandidate] = []
    for box, base_score, source in raw_boxes[:45]:
        refined_boxes = [box, refine_green_plate_box(image, box), refine_light_plate_box(image, box)]
        best_local: PlateCandidate | None = None
        for refined_box in dict.fromkeys(refined_boxes):
            x, y, w, h = clamp_box(refined_box, img_w, img_h)
            crop = image[y : y + h, x : x + w]
            char_boxes, char_mask, char_mode, char_score = segment_characters(crop)
            green_ratio, white_ratio = estimate_plate_color_score(crop)
            ratio = w / float(max(h, 1))
            aspect_score = max(0.0, 1.0 - abs(ratio - PLATE_ASPECT_TARGET) / PLATE_ASPECT_TARGET)
            color_score = min(green_ratio * 2.2, 1.2) + min(white_ratio * 1.1, 0.7)
            if source == "green" and green_ratio < 0.22 and white_ratio < 0.08:
                continue
            if source == "green":
                color_score += 0.4
            if source == "white" and white_ratio > 0.14 and ratio > 1.18:
                color_score += 3.0
            if source in {"edge", "white"} and white_ratio > 0.25 and ratio > 1.15:
                color_score += 5.0
            if source != "green" and green_ratio > 0.45 and white_ratio < 0.08:
                color_score -= 6.0
            if char_mode in {"white_on_green", "light_otsu", "light_adaptive", "mser"} and green_ratio > 0.15:
                color_score += 0.35

            total_score = base_score + char_score * 1.45 + aspect_score * 1.1 + color_score
            y_center = (y + h / 2.0) / img_h
            if y < img_h * 0.03 and h > img_h * 0.12:
                continue
            if y_center < 0.18:
                continue
            if 5 <= len(char_boxes) <= 7:
                total_score += 2.4
            elif len(char_boxes) == 4:
                total_score += 0.5
            elif len(char_boxes) < 4:
                total_score -= 4.0
            if ratio < 1.18 and len(char_boxes) < 5:
                total_score -= 2.0
            if source == "green" and white_ratio < 0.05 and len(char_boxes) < 5:
                total_score -= 8.0

            candidate = PlateCandidate(
                box=(x, y, w, h),
                score=total_score,
                source=f"{source}/{char_mode}",
                char_boxes=char_boxes,
                char_mask=char_mask,
            )
            if best_local is None or candidate.score > best_local.score:
                best_local = candidate

        if best_local is not None:
            scored.append(best_local)

    return max(scored, key=lambda item: item.score) if scored else None


def draw_plate_outputs(
    original: np.ndarray,
    scale: float,
    candidate: PlateCandidate | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, int]], np.ndarray]:
    h, w = original.shape[:2]
    stage1_mask = np.zeros((h, w), dtype=np.uint8)
    foreground = np.zeros_like(original)
    plate_box_vis = original.copy()
    char_box_vis = original.copy()
    char_boxes_original: list[tuple[int, int, int, int]] = []
    plate_crop = np.zeros((1, 1, 3), dtype=np.uint8)

    if candidate is None:
        return stage1_mask, foreground, plate_box_vis, char_box_vis, char_boxes_original, plate_crop

    inv_scale = 1.0 / scale
    x, y, bw, bh = rescale_box(candidate.box, inv_scale)
    x, y, bw, bh = clamp_box((x, y, bw, bh), w, h)
    cv2.rectangle(stage1_mask, (x, y), (x + bw, y + bh), 255, thickness=-1)
    foreground[stage1_mask > 0] = original[stage1_mask > 0]
    cv2.rectangle(plate_box_vis, (x, y), (x + bw, y + bh), (0, 255, 255), 3)

    plate_crop = original[y : y + bh, x : x + bw].copy()
    char_boxes, _, _, _ = segment_characters(plate_crop)
    if len(char_boxes) < 5:
        char_boxes = fallback_even_char_boxes(plate_crop, count=7)
    for cx, cy, cw, ch in char_boxes:
        cx, cy, cw, ch = clamp_box((cx, cy, cw, ch), bw, bh)
        absolute = (x + cx, y + cy, cw, ch)
        char_boxes_original.append(absolute)
        cv2.rectangle(
            char_box_vis,
            (absolute[0], absolute[1]),
            (absolute[0] + absolute[2], absolute[1] + absolute[3]),
            (0, 255, 0),
            2,
        )
    cv2.rectangle(char_box_vis, (x, y), (x + bw, y + bh), (0, 255, 255), 3)

    return stage1_mask, foreground, plate_box_vis, char_box_vis, char_boxes_original, plate_crop


def fallback_even_char_boxes(plate_crop: np.ndarray, count: int = 7) -> list[tuple[int, int, int, int]]:
    h, w = plate_crop.shape[:2]
    if h <= 0 or w <= 0:
        return []

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    light = cv2.inRange(hsv, (0, 0, 70), (180, 210, 255))
    green = cv2.inRange(hsv, (35, 25, 5), (105, 255, 230))
    likely_text = cv2.bitwise_or(dark, cv2.bitwise_and(light, cv2.bitwise_not(green)))
    border_x = max(2, int(w * 0.06))
    border_y = max(2, int(h * 0.12))
    likely_text[:border_y, :] = 0
    likely_text[-border_y:, :] = 0
    likely_text[:, :border_x] = 0
    likely_text[:, -border_x:] = 0

    points = cv2.findNonZero(likely_text)
    if points is not None:
        x, y, bw, bh = cv2.boundingRect(points)
        if bw < w * 0.35 or bh < h * 0.20:
            x, y, bw, bh = int(w * 0.08), int(h * 0.22), int(w * 0.84), int(h * 0.62)
    else:
        x, y, bw, bh = int(w * 0.08), int(h * 0.22), int(w * 0.84), int(h * 0.62)

    y = max(0, y)
    bh = min(h - y, max(int(h * 0.45), bh))
    span_start = max(0, x)
    span_end = min(w, x + bw)
    span = max(1, span_end - span_start)
    step = span / float(count)
    char_w = max(3, int(step * 0.68))
    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(count):
        center = span_start + (idx + 0.5) * step
        cx = int(round(center - char_w / 2.0))
        boxes.append(clamp_box((cx, y, char_w, bh), w, h))
    return boxes


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def process_image(
    image_path: Path,
    output_dir: Path,
    max_width: int,
    write_debug: bool,
) -> dict[str, object]:
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    work, scale = resize_keep_aspect(original, max_width)
    masks = plate_candidate_masks(work)
    candidate = choose_plate_candidate(work, masks)
    stage1_mask, foreground, plate_box_vis, char_box_vis, char_boxes, plate_crop = draw_plate_outputs(
        original, scale, candidate
    )

    name = safe_name(image_path)
    cv2.imwrite(str(output_dir / f"{name}_stage1_mask.png"), stage1_mask)
    cv2.imwrite(str(output_dir / f"{name}_stage1_foreground.jpg"), foreground)
    cv2.imwrite(str(output_dir / f"{name}_plate_box.jpg"), plate_box_vis)
    cv2.imwrite(str(output_dir / f"{name}_char_boxes.jpg"), char_box_vis)
    if plate_crop.size > 3:
        cv2.imwrite(str(output_dir / f"{name}_plate_crop.jpg"), plate_crop)
        _, char_mask, _, _ = segment_characters(plate_crop)
        cv2.imwrite(str(output_dir / f"{name}_char_mask.png"), char_mask)

    if write_debug:
        for mask_name, mask in masks.items():
            debug_mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_dir / f"{name}_debug_{mask_name}.png"), debug_mask)

    plate_box = ""
    candidate_source = ""
    candidate_score = ""
    if candidate is not None:
        plate_box = ",".join(map(str, rescale_box(candidate.box, 1.0 / scale)))
        candidate_source = candidate.source
        candidate_score = f"{candidate.score:.3f}"

    return {
        "image": str(image_path),
        "plate_box": plate_box,
        "char_count": len(char_boxes),
        "char_boxes": ";".join(",".join(map(str, box)) for box in char_boxes),
        "candidate_source": candidate_source,
        "score": candidate_score,
    }


def write_csv(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    rows = list(rows)
    fieldnames = ["image", "plate_box", "char_count", "char_boxes", "candidate_source", "score"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_path)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {input_path}")
        return 1

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        try:
            row = process_image(image_path, output_dir, args.max_width, args.debug)
            rows.append(row)
            print(
                f"[{index}/{len(images)}] {image_path.name}: "
                f"plate={row['plate_box'] or 'not-found'} chars={row['char_count']}"
            )
        except Exception as exc:  # Keep long batch runs moving.
            rows.append(
                {
                    "image": str(image_path),
                    "plate_box": "",
                    "char_count": 0,
                    "char_boxes": "",
                    "candidate_source": "error",
                    "score": str(exc),
                }
            )
            print(f"[{index}/{len(images)}] {image_path.name}: error: {exc}")

    write_csv(rows, output_dir / "results.csv")
    print(f"Results written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
