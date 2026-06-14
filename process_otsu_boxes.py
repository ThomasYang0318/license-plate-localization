import argparse
from pathlib import Path

import cv2

from character_candidates import draw_boxes, find_best_character_boxes, write_result_file


input_folder = Path("Datasets")
grayscale_folder = Path("GrayscaleDatasets")
otsu_folder = Path("OtsuDatasets")
boxed_folder = Path("OtsuBoxedDatasets")
result_folder = Path("OtsuResults")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}


def process_images(args: argparse.Namespace) -> None:
    grayscale_folder.mkdir(exist_ok=True)
    otsu_folder.mkdir(exist_ok=True)
    boxed_folder.mkdir(exist_ok=True)
    result_folder.mkdir(exist_ok=True)

    processed_count = 0

    for image_path in sorted(input_folder.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue

        color_image = cv2.imread(str(image_path))
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if color_image is None or gray_image is None:
            print(f"無法讀取圖片：{image_path}")
            continue

        grayscale_path = grayscale_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(grayscale_path), gray_image)

        threshold_value, otsu_image = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        otsu_path = otsu_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(otsu_path), otsu_image)

        adaptive_image = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            args.adaptive_block_size,
            args.adaptive_c,
        )

        boxes, candidate_source = find_best_character_boxes(
            primary_binary_image=otsu_image,
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
        boxed_image = draw_boxes(color_image, boxes)
        boxed_path = boxed_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(boxed_path), boxed_image)

        result_path = result_folder / f"{image_path.stem}.txt"
        write_result_file(result_path, image_path.stem, boxes)

        processed_count += 1
        print(
            f"{image_path.name} -> {grayscale_path}, {otsu_path}, {boxed_path}, {result_path} "
            f"已完成，Otsu 閾值：{threshold_value:.0f}，候選來源：{candidate_source}，框選 {len(boxes)} 個區域"
        )

    print(f"完成，共處理 {processed_count} 張圖片。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="灰階後使用大津演算法處理，並以 connected component labeling 找字元候選框。")
    parser.add_argument("--min-width", type=int, default=3, help="字元候選框最小寬度，預設為 3。")
    parser.add_argument("--max-width-ratio", type=float, default=0.25, help="字元候選框最大寬度佔圖寬比例，預設為 0.25。")
    parser.add_argument("--min-height", type=int, default=10, help="文字框最小高度，預設為 10。")
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
    args = parser.parse_args()
    if (
        args.min_width < 1
        or args.min_height < 1
        or args.min_area < 1
        or args.min_candidates < 1
    ):
        parser.error("--min-width、--min-height、--min-area、--min-candidates 必須大於或等於 1。")
    if args.adaptive_block_size < 3 or args.adaptive_block_size % 2 == 0:
        parser.error("--adaptive-block-size 必須是大於或等於 3 的奇數。")
    for option_name in ("max_width_ratio", "max_height_ratio", "max_area_ratio", "y_tolerance_ratio", "border_margin_ratio"):
        if not 0 <= getattr(args, option_name) <= 1:
            parser.error(f"--{option_name.replace('_', '-')} 必須介於 0 和 1。")
    if args.min_aspect_ratio <= 0 or args.max_aspect_ratio <= 0:
        parser.error("--min-aspect-ratio 與 --max-aspect-ratio 必須大於 0。")
    if args.min_aspect_ratio > args.max_aspect_ratio:
        parser.error("--min-aspect-ratio 不可大於 --max-aspect-ratio。")
    return args


if __name__ == "__main__":
    process_images(parse_args())
