# License Plate Localization

車牌定位與字元框選專案。現在建議使用 `license_plate_yolo.py` 的兩階段流程：

1. YOLO 偵測車牌位置。
2. 對車牌裁切區做影像增強、牌面範圍修正、字元輪廓切割與框數正規化。

此流程會產生作業需要的第一階段輸出、字元框可視化圖，以及符合提交格式的 txt 檔。

## 快速使用

建立環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

下載模型權重：

```bash
mkdir -p models
curl -L -o models/plate_yolov8.pt \
  https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt
curl -L -o models/char_yolov8.pt \
  https://huggingface.co/MKgoud/License-Plate-Character-Detector/resolve/main/Charcter-LP.pt
```

執行目前使用的快速版本：

```bash
python3 license_plate_yolo.py \
  --input Datasets \
  --output outputs_yolo \
  --plate-conf 0.10 \
  --imgsz 640 \
  --skip-char-yolo

python3 normalize_existing_yolo_outputs.py \
  --output outputs_yolo \
  --txt-output outputs_yolo/submission.txt
```

若要直接輸出成組長學號檔名：

```bash
python3 normalize_existing_yolo_outputs.py \
  --output outputs_yolo \
  --txt-output outputs_yolo/412345678.txt
```

更完整的使用方式與輸出格式請看 [USAGE.md](USAGE.md)。

## 輸出資料夾

執行後本機 `outputs_yolo/` 會包含：

```text
stage1_mask/          車牌為白色前景、其他為背景的二值遮罩
stage1_foreground/    只保留車牌區域的原圖
plate_box/            原圖上畫出車牌框
plate_crop/           車牌裁切圖
plate_processed/      車牌裁切後的增強影像
char_boxes/           原圖上畫出車牌框與字元框
results.csv           每張圖的車牌框、字元框與處理來源
submission.txt        作業提交格式，座標已轉為 1-based
```

本次 400 張影像的輸出統計：

```text
總筆數: 400
字元數分佈: 7 字元 214 張、6 字元 87 張、5 字元 88 張、0 字元 11 張
```

`0 字元` 表示 YOLO 第一階段沒有找到車牌。

GitHub 版本只保留 `outputs_yolo/results.csv` 與 `outputs_yolo/submission.txt`，不放輸出圖片、資料集、venv 或模型權重。完整可視化圖片請用上方指令在本機重新產生。

## 方法摘要

車牌定位使用公開 YOLOv8 車牌模型。字元框選則以車牌裁切區為基礎，先找出白底或綠底牌面，排除車牌後方黑色保險桿與背景，再進行二值化、MSER/輪廓切割、主文字列篩選、寬框拆分與超過 7 框的雜訊合併。

保留的傳統 OpenCV 版本在 `license_plate_detector.py`，可作為不使用 YOLO 的 baseline。

## 參考模型

- Plate YOLO: <https://huggingface.co/Koushim/yolov8-license-plate-detection>
- Character YOLO: <https://huggingface.co/MKgoud/License-Plate-Character-Detector>
