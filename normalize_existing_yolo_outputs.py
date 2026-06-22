#!/usr/bin/env python3
"""Normalize existing YOLO/OpenCV character boxes without rerunning segmentation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from refine_yolo_outputs import (
    clamp_box,
    filter_boxes_to_face,
    find_plate_face_box,
    normalize_char_boxes,
    robust_imread,
    split_wide_boxes,
    write_submission_txt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize existing character boxes and rewrite submission txt."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs_yolo",
        help="Output directory containing results.csv and char_boxes/. Default: outputs_yolo",
    )
    parser.add_argument(
        "--txt-output",
        default="",
        help="Submission txt path. Default: <output>/submission.txt",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=7,
        help="Maximum expected plate character count. Default: 7",
    )
    parser.add_argument(
        "--rewrite-all",
        action="store_true",
        help="Normalize and redraw every detected row instead of only rows above --max-chars.",
    )
    return parser.parse_args()


def parse_boxes(raw_boxes: str) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for item in raw_boxes.split(";"):
        if not item.strip():
            continue
        boxes.append(tuple(int(value) for value in item.split(",")))
    return boxes


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    rows_path = output_dir / "results.csv"
    with rows_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    char_vis_dir = output_dir / "char_boxes"
    char_vis_dir.mkdir(exist_ok=True)

    changed = 0
    for index, row in enumerate(rows, start=1):
        before_count = int(row.get("char_count") or 0)
        if before_count <= args.max_chars and not args.rewrite_all:
            print(f"[{index}/{len(rows)}] {Path(row['image']).name}: keep {before_count}", flush=True)
            continue

        image_path = Path(row["image"])
        image = robust_imread(image_path)
        plate_box = row.get("plate_box", "")
        if image is None or not plate_box:
            row["char_count"] = "0"
            row["char_boxes"] = ""
            row["recognized"] = ""
            row["source"] = "no_plate"
            continue

        px, py, pw, ph = [int(value) for value in plate_box.split(",")]
        px, py, pw, ph = clamp_box((px, py, pw, ph), image.shape[1], image.shape[0])
        crop = image[py : py + ph, px : px + pw]
        face_box = find_plate_face_box(crop)

        crop_boxes = [
            (x - px, y - py, w, h)
            for x, y, w, h in parse_boxes(row.get("char_boxes", ""))
        ]
        crop_boxes = filter_boxes_to_face(crop_boxes, face_box)
        crop_boxes = split_wide_boxes(crop_boxes, face_box)
        crop_boxes = normalize_char_boxes(crop_boxes, face_box, max_chars=args.max_chars)

        absolute_boxes = [(px + x, py + y, w, h) for x, y, w, h in crop_boxes]
        row["char_count"] = str(len(absolute_boxes))
        row["recognized"] = "?" * len(absolute_boxes)
        row["char_boxes"] = ";".join(",".join(map(str, box)) for box in absolute_boxes)
        row["source"] = "yolo_plate_normalized_chars"
        if len(absolute_boxes) != before_count:
            changed += 1

        vis = image.copy()
        cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 255, 255), 3)
        for bx, by, bw, bh in absolute_boxes:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.imwrite(str(char_vis_dir / f"{image_path.stem}.jpg"), vis)
        print(f"[{index}/{len(rows)}] {image_path.name}: {before_count}->{len(absolute_boxes)}", flush=True)

    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image", "plate_box", "char_count", "recognized", "char_boxes", "source", "plate_conf"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    txt_path = Path(args.txt_output) if args.txt_output else output_dir / "submission.txt"
    write_submission_txt(rows, txt_path)
    print(f"Normalized {changed} rows; wrote {rows_path} and {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
