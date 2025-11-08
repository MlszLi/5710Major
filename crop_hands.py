# -*- coding: utf-8 -*-
"""
crop_hands.py
批量：图片/视频 -> YOLO手部检测 -> 扩框 -> 统一尺寸裁剪 -> 保存
依赖：ultralytics, opencv-python, numpy, tqdm
安装：pip install ultralytics opencv-python numpy tqdm
"""
import argparse
import os
import sys
from pathlib import Path
import csv

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    print("❌ 请先安装依赖：pip install ultralytics opencv-python numpy tqdm")
    raise

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_images_recursive(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield p

def letterbox_square(img, size=256, color=(128,128,128)):
    """等比缩放+灰边填充成正方形"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def expand_box(xyxy, W, H, r=0.15):
    """xyxy 扩框比例 r，并裁剪到图像范围内"""
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    dx, dy = int(w * r), int(h * r)
    x1 = max(0, int(x1) - dx)
    y1 = max(0, int(y1) - dy)
    x2 = min(W, int(x2) + dx)
    y2 = min(H, int(y2) + dy)
    return x1, y1, x2, y2

def save_crop(img, box, out_dir: Path, stem: str, idx: int, size: int):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = expand_box(box, W, H, r=args.expand)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = letterbox_square(crop, size=size)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{idx}.png"
    cv2.imwrite(str(out_path), crop)
    return out_path, (x1, y1, x2, y2)

def load_model(model_path: str, device_arg: str):
    model = YOLO(model_path)
    # 让 ultralytics 用GPU（如果可用）
    device = 0 if (device_arg == "auto" and torch.cuda.is_available()) else device_arg
    return model, device

def process_image_folder(model, device, src_dir: Path, out_dir: Path, args):
    index_csv = out_dir / "crops_index.csv"
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src_image", "crop_path", "x1", "y1", "x2", "y2", "conf"])

        imgs = list(list_images_recursive(src_dir))
        pbar = tqdm(imgs, desc="Processing images")
        for img_path in pbar:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            H, W = im.shape[:2]
            res = model.predict(
                source=im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                verbose=False, device=device, max_det=args.max_det
            )[0]

            if len(res.boxes) == 0:
                continue

            # 按置信度排序（高->低）
            confs = res.boxes.conf.cpu().numpy()
            order = np.argsort(-confs)
            for k, i in enumerate(order):
                b = res.boxes[i]
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf.cpu().item())
                out = save_crop(im, xyxy, out_dir, img_path.stem, k, args.size)
                if out is None:
                    continue
                out_path, (x1, y1, x2, y2) = out
                writer.writerow([str(img_path), str(out_path), x1, y1, x2, y2, f"{conf:.4f}"])

def process_video(model, device, video_path: Path, out_dir: Path, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        return
    index_csv = out_dir / "crops_index.csv"
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src_video", "frame_idx", "crop_path", "x1", "y1", "x2", "y2", "conf"])

        frame_idx = 0
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % args.frame_step != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                H, W = frame.shape[:2]
                res = model.predict(
                    source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                    verbose=False, device=device, max_det=args.max_det
                )[0]

                if len(res.boxes) > 0:
                    confs = res.boxes.conf.cpu().numpy()
                    order = np.argsort(-confs)
                    for k, i in enumerate(order):
                        b = res.boxes[i]
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        conf = float(b.conf.cpu().item())
                        out = save_crop(frame, xyxy, out_dir, f"{video_path.stem}_f{frame_idx}", k, args.size)
                        if out is None:
                            continue
                        out_path, (x1, y1, x2, y2) = out
                        writer.writerow([str(video_path), frame_idx, str(out_path), x1, y1, x2, y2, f"{conf:.4f}"])

                frame_idx += 1
                pbar.update(1)
    cap.release()

def parse_args():
    ap = argparse.ArgumentParser(description="批量裁剪手部ROI（图片/视频）")
    ap.add_argument("--model", required=True, help="Root\yolo\best.pt")
    ap.add_argument("--source", required=True, help="Root\yolo\rgb_00051.png")
    ap.add_argument("--out", required=True, help="Root\yolo")
    ap.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    ap.add_argument("--max-det", type=int, default=5, help="每张图最多保留手的数量")
    ap.add_argument("--expand", type=float, default=0.15, help="bbox四周扩张比例")
    ap.add_argument("--size", type=int, default=256, help="输出裁剪图统一尺寸（正方形）")
    ap.add_argument("--device", default="auto", help="'auto' 或 0/1/cpu")
    ap.add_argument("--frame-step", type=int, default=1, help="视频取帧间隔（1=每帧）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model, device = load_model(args.model, args.device)

    src = Path(args.source)
    out_dir = Path(args.out)

    print(f"➡️  使用设备: {device}")
    print(f"➡️  模型: {args.model}")
    print(f"➡️  源: {src}")
    print(f"➡️  输出: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        process_image_folder(model, device, src, out_dir, args)
    elif src.is_file():
        process_video(model, device, src, out_dir, args)
    else:
        print("❌ --source 既不是文件夹也不是文件")
        sys.exit(1)

    print("✅ 完成！裁剪记录见：", out_dir / "crops_index.csv")
