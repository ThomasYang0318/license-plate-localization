import argparse
from pathlib import Path

import cv2
import numpy as np

from character_candidates import (
    draw_boxes,
    find_best_character_boxes,
    find_character_boxes,
    find_plate_guided_character_boxes,
    mask_to_roi,
    refine_character_boxes,
    write_result_file,
)
from image_io import read_image_pair
from plate_detection import find_yolo_plate_roi, load_yolo_model


input_folder = Path("Datasets")
output_folder = Path("outputs")
grayscale_folder = output_folder / "grayscale"
sobel_folder = output_folder / "sobel" / "binary"
boxed_folder = output_folder / "sobel" / "boxed"
result_folder = output_folder / "sobel" / "results"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}


def apply_lower_center_blackhat(gray_image, kernel_width: int, kernel_height: int):
    image_height, image_width = gray_image.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    blackhat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    _, binary_image = cv2.threshold(blackhat_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    masked_image = np.zeros_like(binary_image)
    y_start = int(image_height * 0.55)
    x_start = int(image_width * 0.20)
    x_end = int(image_width * 0.65)
    masked_image[y_start:, x_start:x_end] = binary_image[y_start:, x_start:x_end]
    return masked_image


def median_center_y_ratio(boxes, image_height: int):
    if not boxes:
        return 0.0
    centers_y = sorted(y + height / 2 for _, y, _, height in boxes)
    return centers_y[len(centers_y) // 2] / image_height


def apply_sobel(gray_image, kernel_size: int):
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_image = cv2.convertScaleAbs(magnitude)
    _, sobel_binary = cv2.threshold(sobel_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return sobel_binary

def process_images(args: argparse.Namespace) -> None:
    grayscale_folder.mkdir(parents=True, exist_ok=True)
    sobel_folder.mkdir(parents=True, exist_ok=True)
    boxed_folder.mkdir(parents=True, exist_ok=True)
    result_folder.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    only_names = {Path(name).stem for name in args.only} if args.only else None
    yolo_model = load_yolo_model(args.yolo_model) if args.yolo_model else None

    for image_path in sorted(input_folder.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue
        if only_names is not None and image_path.stem not in only_names:
            continue

        color_image, gray_image, read_error = read_image_pair(image_path, args.image_read_timeout)
        if color_image is None or gray_image is None:
            print(f"無法讀取圖片：{image_path} ({read_error})", flush=True)
            continue

        grayscale_path = grayscale_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(grayscale_path), gray_image)

        sobel_image = apply_sobel(gray_image, kernel_size=args.sobel_kernel_size)
        sobel_path = sobel_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(sobel_path), sobel_image)

        adaptive_image = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            args.adaptive_block_size,
            args.adaptive_c,
        )

        boxes = []
        candidate_source = ""
        if yolo_model is not None:
            yolo_roi = find_yolo_plate_roi(
                yolo_model,
                color_image,
                confidence_threshold=args.yolo_confidence,
                padding_ratio=args.yolo_padding_ratio,
            )
            if yolo_roi is not None:
                boxes, _ = find_best_character_boxes(
                    primary_binary_image=mask_to_roi(sobel_image, yolo_roi),
                    fallback_binary_image=mask_to_roi(adaptive_image, yolo_roi),
                    min_candidates=args.min_candidates,
                    min_width=args.min_width,
                    max_width_ratio=args.max_width_ratio,
                    min_height=args.min_height,
                    max_height_ratio=args.max_height_ratio,
                    min_area=args.min_area,
                    max_area_ratio=args.max_area_ratio,
                    min_aspect_ratio=args.min_aspect_ratio,
                    max_aspect_ratio=args.max_aspect_ratio,
                    y_tolerance_ratio=args.y_tolerance_ratio,
                    border_margin_ratio=args.border_margin_ratio,
                )
                candidate_source = "yolo_plate"

        if not boxes:
            boxes, candidate_source = find_plate_guided_character_boxes(
                gray_image=gray_image,
                primary_binary_image=sobel_image,
                fallback_binary_image=adaptive_image,
                min_candidates=args.min_candidates,
                min_width=args.min_width,
                max_width_ratio=args.max_width_ratio,
                min_height=args.min_height,
                max_height_ratio=args.max_height_ratio,
                min_area=args.min_area,
                max_area_ratio=args.max_area_ratio,
                min_aspect_ratio=args.min_aspect_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                y_tolerance_ratio=args.y_tolerance_ratio,
                border_margin_ratio=args.border_margin_ratio,
            )

        if not boxes:
            boxes, candidate_source = find_best_character_boxes(
                primary_binary_image=sobel_image,
                fallback_binary_image=adaptive_image,
                min_candidates=args.min_candidates,
                min_width=args.min_width,
                max_width_ratio=args.max_width_ratio,
                min_height=args.min_height,
                max_height_ratio=args.max_height_ratio,
                min_area=args.min_area,
                max_area_ratio=args.max_area_ratio,
                min_aspect_ratio=args.min_aspect_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                y_tolerance_ratio=args.y_tolerance_ratio,
                border_margin_ratio=args.border_margin_ratio,
            )
        blackhat_image = apply_lower_center_blackhat(
            gray_image,
            kernel_width=args.blackhat_kernel_width,
            kernel_height=args.blackhat_kernel_height,
        )
        blackhat_boxes = find_character_boxes(
            blackhat_image,
            min_width=args.min_width,
            max_width_ratio=args.max_width_ratio,
            min_height=args.min_height,
            max_height_ratio=args.max_height_ratio,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            min_aspect_ratio=0.08,
            max_aspect_ratio=max(args.max_aspect_ratio, 1.8),
            y_tolerance_ratio=args.y_tolerance_ratio,
            border_margin_ratio=args.border_margin_ratio,
        )
        if (
            len(blackhat_boxes) >= args.min_candidates
            and (
                (
                    candidate_source not in {"plate_roi", "yolo_plate"}
                    and median_center_y_ratio(boxes, gray_image.shape[0]) < args.upper_result_y_ratio
                )
                or (
                    candidate_source == "plate_roi"
                    and len(boxes) <= args.min_candidates
                    and len(blackhat_boxes) > len(boxes)
                )
            )
            and median_center_y_ratio(blackhat_boxes, gray_image.shape[0]) >= args.lower_blackhat_y_ratio
        ):
            boxes = blackhat_boxes
            candidate_source = "blackhat"

        boxes = refine_character_boxes(
            boxes,
            min_count=args.expected_min_chars,
            max_count=args.expected_max_chars,
        )

        boxed_image = draw_boxes(color_image, boxes)
        boxed_path = boxed_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(boxed_path), boxed_image)

        result_path = result_folder / f"{image_path.stem}.txt"
        write_result_file(result_path, image_path.stem, boxes)

        processed_count += 1
        print(
            f"{image_path.name} -> {grayscale_path}, {sobel_path}, {boxed_path}, {result_path} "
            f"已完成，Sobel kernel：{args.sobel_kernel_size}x{args.sobel_kernel_size}，候選來源：{candidate_source}，框選 {len(boxes)} 個區域",
            flush=True,
        )

    print(f"完成，共處理 {processed_count} 張圖片。", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="灰階後使用 Sobel 處理，並以 connected component labeling 找字元候選框。")
    parser.add_argument("--sobel-kernel-size", type=int, default=3, help="Sobel kernel 尺寸，預設為 3。")
    parser.add_argument("--min-width", type=int, default=3, help="字元候選框最小寬度，預設為 3。")
    parser.add_argument("--max-width-ratio", type=float, default=0.25, help="字元候選框最大寬度佔圖寬比例，預設為 0.25。")
    parser.add_argument("--min-height", type=int, default=10, help="字元候選框最小高度，預設為 10。")
    parser.add_argument("--max-height-ratio", type=float, default=0.9, help="字元候選框最大高度佔圖高比例，預設為 0.9。")
    parser.add_argument("--min-area", type=int, default=20, help="字元候選框最小面積，預設為 20。")
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.08,
        help="字元候選框最大面積佔整張圖比例，預設為 0.08。",
    )
    parser.add_argument("--min-aspect-ratio", type=float, default=0.12, help="字元候選框最小寬高比，預設為 0.12。")
    parser.add_argument("--max-aspect-ratio", type=float, default=1.2, help="字元候選框最大寬高比，預設為 1.2。")
    parser.add_argument("--y-tolerance-ratio", type=float, default=0.025, help="保留同一文字列的 y 中心容許比例，預設為 0.025。")
    parser.add_argument("--border-margin-ratio", type=float, default=0.01, help="排除靠近影像邊框物件的邊界比例，預設為 0.01。")
    parser.add_argument("--min-candidates", type=int, default=4, help="primary 方法少於此候選數時改用 adaptive fallback，預設為 4。")
    parser.add_argument("--adaptive-block-size", type=int, default=31, help="adaptive threshold block size，必須為奇數，預設為 31。")
    parser.add_argument("--adaptive-c", type=int, default=9, help="adaptive threshold C 值，預設為 9。")
    parser.add_argument("--blackhat-kernel-width", type=int, default=15, help="lower-center black-hat kernel 寬度，預設為 15。")
    parser.add_argument("--blackhat-kernel-height", type=int, default=5, help="lower-center black-hat kernel 高度，預設為 5。")
    parser.add_argument("--upper-result-y-ratio", type=float, default=0.45, help="primary 結果中心低於此 y 比例時允許 lower black-hat 接管，預設為 0.45。")
    parser.add_argument("--lower-blackhat-y-ratio", type=float, default=0.55, help="black-hat 結果中心需高於此 y 比例才可接管，預設為 0.55。")
    parser.add_argument("--expected-min-chars", type=int, default=6, help="預期最少字元數，預設為 6。")
    parser.add_argument("--expected-max-chars", type=int, default=7, help="預期最多字元數，預設為 7。")
    parser.add_argument("--only", nargs="*", default=None, help="只處理指定影像編號或檔名，例如 --only 003 005。")
    parser.add_argument("--yolo-model", type=Path, default=None, help="YOLO 車牌偵測模型路徑，例如 models/license_plate.pt。")
    parser.add_argument("--yolo-confidence", type=float, default=0.25, help="YOLO 車牌偵測信心門檻，預設為 0.25。")
    parser.add_argument("--yolo-padding-ratio", type=float, default=0.08, help="YOLO 車牌框外擴比例，預設為 0.08。")
    parser.add_argument("--image-read-timeout", type=float, default=10.0, help="單張圖片讀取逾時秒數，0 代表不啟用，預設為 10。")
    args = parser.parse_args()
    if args.sobel_kernel_size not in {1, 3, 5, 7}:
        parser.error("--sobel-kernel-size 必須是 1、3、5、7 其中之一。")
    if (
        args.min_width < 1
        or args.min_height < 1
        or args.min_area < 1
        or args.min_candidates < 1
        or args.blackhat_kernel_width < 1
        or args.blackhat_kernel_height < 1
        or args.expected_min_chars < 1
        or args.expected_max_chars < 1
        or args.image_read_timeout < 0
    ):
        parser.error("--min-width、--min-height、--min-area、--min-candidates、--blackhat-kernel-width、--blackhat-kernel-height、--expected-min-chars、--expected-max-chars 必須大於或等於 1，--image-read-timeout 必須大於或等於 0。")
    if args.expected_min_chars > args.expected_max_chars:
        parser.error("--expected-min-chars 不可大於 --expected-max-chars。")
    if args.yolo_model is not None and not args.yolo_model.exists():
        parser.error(f"--yolo-model 指定的模型不存在：{args.yolo_model}")
    if args.adaptive_block_size < 3 or args.adaptive_block_size % 2 == 0:
        parser.error("--adaptive-block-size 必須是大於或等於 3 的奇數。")
    for option_name in ("max_width_ratio", "max_height_ratio", "max_area_ratio", "y_tolerance_ratio", "border_margin_ratio", "upper_result_y_ratio", "lower_blackhat_y_ratio", "yolo_confidence", "yolo_padding_ratio"):
        if not 0 <= getattr(args, option_name) <= 1:
            parser.error(f"--{option_name.replace('_', '-')} 必須介於 0 和 1。")
    if args.min_aspect_ratio <= 0 or args.max_aspect_ratio <= 0:
        parser.error("--min-aspect-ratio 與 --max-aspect-ratio 必須大於 0。")
    if args.min_aspect_ratio > args.max_aspect_ratio:
        parser.error("--min-aspect-ratio 不可大於 --max-aspect-ratio。")
    return args


if __name__ == "__main__":
    process_images(parse_args())
