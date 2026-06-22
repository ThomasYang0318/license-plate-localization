# 使用說明

## 1. 專案結構

```text
license_plate_yolo.py              YOLO 車牌定位 + 影像處理字元框選主程式
normalize_existing_yolo_outputs.py 修正既有輸出的字元框數，並重新產生提交 txt
refine_yolo_outputs.py             重新從 YOLO 車牌框做字元後處理的工具
license_plate_detector.py          傳統 OpenCV baseline
outputs_yolo/                      GitHub 內只保留 results.csv 與 submission.txt
```

## 2. 安裝環境

建議使用 Python 3.10 以上：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 準備模型

`license_plate_yolo.py` 預設讀取以下檔案：

```text
models/plate_yolov8.pt
models/char_yolov8.pt
```

下載指令：

```bash
mkdir -p models
curl -L -o models/plate_yolov8.pt \
  https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt
curl -L -o models/char_yolov8.pt \
  https://huggingface.co/MKgoud/License-Plate-Character-Detector/resolve/main/Charcter-LP.pt
```

目前實測字元 YOLO 在本機匯入較慢，因此本版預設建議使用 `--skip-char-yolo`，也就是 YOLO 只做車牌定位，字元框由影像處理完成。

## 4. 執行整批影像

輸入影像放在 `Datasets/`：

```bash
python3 license_plate_yolo.py \
  --input Datasets \
  --output outputs_yolo \
  --plate-conf 0.10 \
  --imgsz 640 \
  --skip-char-yolo
```

接著做字元框正規化，避免同一字元被拆成多框，或把車牌邊緣/黑色背景誤當字元：

```bash
python3 normalize_existing_yolo_outputs.py \
  --output outputs_yolo \
  --txt-output outputs_yolo/submission.txt
```

若提交檔要用組長學號命名：

```bash
python3 normalize_existing_yolo_outputs.py \
  --output outputs_yolo \
  --txt-output outputs_yolo/412345678.txt
```

## 5. 單張或少量測試

只跑前 10 張：

```bash
python3 license_plate_yolo.py \
  --input Datasets \
  --output outputs_yolo_test \
  --plate-conf 0.10 \
  --imgsz 640 \
  --skip-char-yolo \
  --limit 10
```

指定單張圖：

```bash
python3 license_plate_yolo.py \
  --input Datasets/006.jpg \
  --output outputs_single \
  --plate-conf 0.10 \
  --imgsz 640 \
  --skip-char-yolo
```

## 6. 輸出內容

```text
本機執行後的 `outputs_yolo/`：
  stage1_mask/          車牌處為白色前景，其餘為黑色背景
  stage1_foreground/    只保留車牌區域的原圖
  plate_box/            原圖上畫出 YOLO 車牌框
  plate_crop/           車牌裁切圖
  plate_processed/      車牌增強後影像
  char_boxes/           原圖上畫出字元 BBox
  results.csv           偵測結果總表
  submission.txt        作業提交格式
```

GitHub 版本不放輸出圖片；repo 內只保留：

```text
outputs_yolo/results.csv
outputs_yolo/submission.txt
```

## 7. 提交 txt 格式

輸出座標為原圖座標，且已轉成題目要求的 `(1,1)` 起點。

```text
圖片檔名，不含副檔名
偵測到的字元數量
x1 y1 width height
x1 y1 width height
...
```

範例：

```text
004
6
978 465 15 46
995 457 14 48
1013 457 14 48
1029 452 17 44
1047 448 10 44
1061 440 28 49
```

## 8. 目前這批輸出狀態

本次 `outputs_yolo/results.csv` 共 400 筆：

```text
7 字元: 214
6 字元: 87
5 字元: 88
0 字元: 11
```

沒有 8 字元以上的輸出。`0 字元` 代表第一階段未偵測到車牌。

## 9. 常用參數

```text
--plate-conf       車牌 YOLO 信心門檻，預設 0.20，本批使用 0.10
--imgsz            車牌 YOLO 輸入尺寸，本批使用 640
--skip-char-yolo   不跑字元 YOLO，改用影像處理切字元
--txt-output       指定提交 txt 檔路徑
--limit            只處理前 N 張，方便測試
```
