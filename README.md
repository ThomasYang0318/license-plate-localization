# License Plate Localization

本專案使用傳統影像處理方法進行車牌字元候選框偵測，提供兩種流程：

1. 灰階 + 大津演算法 Otsu
2. 灰階 + Sobel 邊緣偵測

兩種方法都會輸出中間處理影像、框選後影像，以及符合測試格式的座標文字檔。

## 環境需求

```bash
pip install -r requirements.txt
```

目前主要套件：

```text
opencv-python
```

## 輸入資料

請將測試影像放在：

```text
Datasets/
```

支援副檔名：

```text
.jpg .jpeg .png .bmp
```

目前範例輸入：

```text
Datasets/001.jpg
Datasets/002.jpg
Datasets/004.jpg
Datasets/020.jpg
```

## 使用方式

執行大津演算法流程：

```bash
python3 process_otsu_boxes.py
```

執行 Sobel 流程：

```bash
python3 process_sobel_boxes.py
```

兩支程式可以分開執行，也可以依序執行來比較兩種方法的輸出結果。

## 輸出資料夾

所有輸出集中在 `outputs/`：

```text
outputs/
  grayscale/
    001.png
    ...
  otsu/
    binary/
    boxed/
    results/
  sobel/
    binary/
    boxed/
    results/
```

各資料夾用途：

```text
outputs/grayscale/       灰階影像
outputs/otsu/binary/     大津二值化影像
outputs/otsu/boxed/      大津方法框選結果
outputs/otsu/results/    大津方法座標輸出
outputs/sobel/binary/    Sobel 二值化影像
outputs/sobel/boxed/     Sobel 方法框選結果
outputs/sobel/results/   Sobel 方法座標輸出
```

## 座標輸出格式

每張輸入影像會輸出一個 `.txt` 檔。

範例：

```text
020
6
516 241 15 28
532 241 16 28
549 242 16 28
566 243 15 27
588 243 15 27
604 244 16 27
```

格式說明：

```text
照片檔名，不含副檔名
偵測到的字元候選框數量
x1 y1 width height
x1 y1 width height
...
```

座標規則：

- `x1, y1` 是 BBox 左上角座標。
- 座標以原圖左上角第一個像素 `(1,1)` 為起點。
- `width` 是 BBox 寬度。
- `height` 是 BBox 高度。
- 輸出已依照 `x1` 由左到右排序。

## 模型框架

本專案不是使用深度學習模型，而是使用傳統影像處理規則建立字元候選框偵測流程。

共用候選框偵測邏輯在：

```text
character_candidates.py
```

整體框架：

```text
輸入影像
-> 灰階
-> 影像處理方法
   -> Otsu
   -> Sobel
-> connected component labeling
-> region properties
   -> x, y, width, height, area, aspect ratio
-> 規則篩選
   -> 寬度
   -> 高度
   -> 面積
   -> 長寬比
   -> y 位置
   -> 邊界位置
-> 選出最像同一列文字的候選框
-> 用 x 方向間距排除螺絲、破折號、邊框線與背景雜訊
-> 依 x 座標排序
-> 輸出 boxed image 與 txt 座標檔
```

## 大津流程

程式：

```text
process_otsu_boxes.py
```

流程：

```text
原圖
-> 灰階
-> Otsu thresholding
-> 若候選框過少，使用 adaptive threshold fallback
-> connected component labeling
-> 字元候選框篩選
-> 輸出結果
```

## Sobel 流程

程式：

```text
process_sobel_boxes.py
```

流程：

```text
原圖
-> 灰階
-> Gaussian blur
-> Sobel x/y gradient
-> gradient magnitude
-> Otsu thresholding
-> 若候選框過少，使用 adaptive threshold fallback
-> connected component labeling
-> 字元候選框篩選
-> 輸出結果
```

## 重要參數

兩種方法都有以下候選框篩選參數：

```bash
--min-width
--max-width-ratio
--min-height
--max-height-ratio
--min-area
--max-area-ratio
--min-aspect-ratio
--max-aspect-ratio
--y-tolerance-ratio
--border-margin-ratio
--min-candidates
--adaptive-block-size
--adaptive-c
```

Sobel 額外參數：

```bash
--sobel-kernel-size
```

查看參數說明：

```bash
python3 process_otsu_boxes.py --help
python3 process_sobel_boxes.py --help
```

## 參考文獻

1. N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," IEEE Transactions on Systems, Man, and Cybernetics, 1979.
2. I. Sobel and G. Feldman, "A 3x3 Isotropic Gradient Operator for Image Processing," Stanford Artificial Intelligence Project, 1968.
3. R. C. Gonzalez and R. E. Woods, Digital Image Processing, Pearson.
4. OpenCV documentation: thresholding, Sobel derivatives, connected components, and image I/O.
5. `References/Automatic_Vehicle_License_Plate_Recognition_System_Based_on_Image_Processing_and_Template_Matching_Approach.pdf`
