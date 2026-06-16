import argparse
from pathlib import Path


def export_results(results_folder: Path, output_path: Path, image_folder: Path) -> None:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_names = {
        image_path.stem
        for image_path in image_folder.iterdir()
        if image_path.suffix.lower() in image_extensions
    }
    result_files = sorted(
        result_file
        for result_file in results_folder.glob("*.txt")
        if result_file.stem in image_names
    )
    output_lines = []
    exported_count = 0

    for result_file in result_files:
        if result_file.resolve() == output_path.resolve():
            continue

        lines = result_file.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue

        output_lines.extend(lines)
        exported_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"已輸出 {output_path}，共彙整 {exported_count} 個結果檔。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="將每張影像的座標結果彙整成單一 txt 檔。")
    parser.add_argument(
        "--results-folder",
        type=Path,
        default=Path("outputs/otsu/results"),
        help="每張影像結果 txt 所在資料夾，預設為 outputs/otsu/results。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/otsu/411285003.txt"),
        help="彙整後輸出檔案，預設為 outputs/otsu/411285003.txt。",
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path("Datasets"),
        help="用來決定要匯出哪些影像編號的資料夾，預設為 Datasets。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_results(args.results_folder, args.output, args.image_folder)
