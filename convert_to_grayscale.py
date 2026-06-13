from pathlib import Path

import cv2

# 原始圖片資料夾
input_folder = Path("Datasets")

# 灰階圖片要存放的新資料夾
output_folder = Path("GrayscaleDatasets")

# 要處理的圖片副檔名
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}


# 如果輸出資料夾不存在，就自動建立
output_folder.mkdir(exist_ok=True)

converted_count = 0

# 讀取 Datasets 資料夾裡的每一個檔案
for image_path in input_folder.iterdir():
    # 只處理圖片檔，其他檔案會略過
    if image_path.suffix.lower() not in image_extensions:
        continue

    # 直接用灰階模式讀取圖片
    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # 如果圖片讀取失敗，就跳過這張
    if gray_image is None:
        print(f"無法讀取圖片：{image_path}")
        continue

    # 使用原本的檔名，存到新的資料夾
    output_path = output_folder / image_path.name
    cv2.imwrite(str(output_path), gray_image)

    converted_count += 1

print(f"完成，共轉換 {converted_count} 張圖片。")
