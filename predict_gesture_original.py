# -*- coding: utf-8 -*-
import sys, os, time
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QProgressBar, QLineEdit, QMessageBox
)

# ==== 依赖：YOLO服务 & 分类器 ====
import torch
# ▼ 新版对接：从新版 webcam_detect_and_crop.py 引入 HandCamService
from webcam_detect_and_crop import HandCamService
from predict_gesture import load_model, predict_bgr, CLASSES  # 你已有的分类器

def cv_to_qpixmap(img_bgr):
    if img_bgr is None: return QPixmap()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h,w,c = rgb.shape
    qimg = QImage(rgb.data, w, h, c*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def letterbox_square(img, size=480, color=(20,20,20)):
    h, w = img.shape[:2]
    s = min(size/h, size/w)
    nh, nw = int(h*s), int(w*s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size,size,3), color, np.uint8)
    top = (size-nh)//2; left=(size-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

# 保留：如果以后你仍需本地裁剪，可用到；当前用不到也不删，避免“无关改动”
def expand_box(xyxy, W, H, r=0.45):
    x1, y1, x2, y2 = map(int, xyxy)
    w, h = x2-x1, y2-y1
    dx, dy = int(w*r), int(h*r)
    x1 = max(0, x1-dx); y1 = max(0, y1-dy)
    x2 = min(W, x2+dx); y2 = min(H, y2+dy)
    return x1,y1,x2,y2

class LiveGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live: Camera → YOLO(Service) → Gesture")
        self.resize(1280, 760)

        # 路径输入
        self.ed_yolo_w = QLineEdit()
        self.ed_cls_w  = QLineEdit()
        btn_yolo_w = QPushButton("选 YOLO 权重(.pt)"); btn_yolo_w.clicked.connect(lambda: self.pick(self.ed_yolo_w, "PyTorch (*.pt)"))
        btn_cls_w  = QPushButton("选 分类权重(.pt)");   btn_cls_w.clicked.connect(lambda: self.pick(self.ed_cls_w,  "PyTorch (*.pt)"))

        self.btn_start = QPushButton("启动摄像头"); self.btn_start.clicked.connect(self.on_start)
        self.btn_stop  = QPushButton("停止");       self.btn_stop.clicked.connect(self.on_stop); self.btn_stop.setEnabled(False)

        top = QGridLayout()
        top.addWidget(QLabel("YOLO权重:"),0,0); top.addWidget(self.ed_yolo_w,0,1); top.addWidget(btn_yolo_w,0,2)
        top.addWidget(QLabel("分类权重:"),1,0);  top.addWidget(self.ed_cls_w,1,1);  top.addWidget(btn_cls_w,1,2)
        top.addWidget(self.btn_start,2,0); top.addWidget(self.btn_stop,2,2)

        # 画面
        self.lbl_cam  = QLabel("Camera"); self.lbl_cam.setAlignment(Qt.AlignCenter);  self.lbl_cam.setMinimumSize(560,420); self.lbl_cam.setStyleSheet("background:#111;color:#aaa;")
        self.lbl_crop = QLabel("Hand ROI"); self.lbl_crop.setAlignment(Qt.AlignCenter); self.lbl_crop.setMinimumSize(560,420); self.lbl_crop.setStyleSheet("background:#111;color:#aaa;")
        imgs = QHBoxLayout(); imgs.addWidget(self.lbl_cam,1); imgs.addWidget(self.lbl_crop,1)

        # 结果
        self.lbl_top1 = QLabel("—"); self.pb1 = QProgressBar(); self.pb1.setRange(0,100)
        self.lbl_top2 = QLabel("—"); self.pb2 = QProgressBar(); self.pb2.setRange(0,100)
        self.lbl_top3 = QLabel("—"); self.pb3 = QProgressBar(); self.pb3.setRange(0,100)
        resg = QGroupBox("Top-K")
        grid = QGridLayout()
        grid.addWidget(QLabel("Top-1:"),0,0); grid.addWidget(self.lbl_top1,0,1); grid.addWidget(self.pb1,0,2)
        grid.addWidget(QLabel("Top-2:"),1,0); grid.addWidget(self.lbl_top2,1,1); grid.addWidget(self.pb2,1,2)
        grid.addWidget(QLabel("Top-3:"),2,0); grid.addWidget(self.lbl_top3,2,1); grid.addWidget(self.pb3,2,2)
        resg.setLayout(grid)

        central = QWidget(); v = QVBoxLayout(central)
        v.addLayout(top); v.addSpacing(6); v.addLayout(imgs); v.addSpacing(6); v.addWidget(resg)
        self.setCentralWidget(central)

        # 状态
        self.timer = QTimer(self); self.timer.timeout.connect(self.on_tick)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ▼ 新增：服务实例与节流
        self.service = None
        self._last_detect_t = 0.0
        self.detect_interval_sec = 0.35  # 每隔 ~350ms 触发一次 detect_once()

        # 分类器
        self.cls_model = None

    def pick(self, line: QLineEdit, filt: str):
        p, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if p: line.setText(p)

    def on_start(self):
        if not Path(self.ed_yolo_w.text()).exists() or not Path(self.ed_cls_w.text()).exists():
            QMessageBox.information(self,"提示","请先选择 YOLO 与 分类权重 .pt"); return

        # 1) 加载分类器（只加载一次）
        if self.cls_model is None:
            self.statusBar().showMessage(f"Loading classifier on {self.device} …")
            self.cls_model = load_model(self.ed_cls_w.text(), self.device, num_classes=len(CLASSES))

        # 2) 启动 HandCamService（摄像头 + YOLO 懒加载）
        if self.service is None:
            self.statusBar().showMessage("Starting camera service …")
            self.service = HandCamService(
                model_path=self.ed_yolo_w.text(),
                # 以下参数与新版服务保持一致或使用你期望的默认值
                imgsz=640,
                conf_thr=0.28,
                iou_thr=0.55,
                expand_r=0.30,             # 截图范围更大
                target_size=(300, 300),    # 服务保存的 ROI 尺寸
                pad_color=(255, 255, 255), # 白边
                preview_width=1280,
                preview_height=720,
            )
            self.service.start()
            self._last_detect_t = 0.0

        # 3) 开计时器（约 20 fps）
        self.timer.start(50)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Running …")

    def on_stop(self):
        self.timer.stop()
        if self.service is not None:
            self.service.stop()
            self.service = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def set_topk(self, items):
        def set_row(lbl, pb, tup):
            if tup: lbl.setText(f"{tup[0]} ({tup[1]:.3f})"); pb.setValue(int(round(tup[1]*100)))
            else:   lbl.setText("—"); pb.setValue(0)
        set_row(self.lbl_top1, self.pb1, items[0] if len(items)>0 else None)
        set_row(self.lbl_top2, self.pb2, items[1] if len(items)>1 else None)
        set_row(self.lbl_top3, self.pb3, items[2] if len(items)>2 else None)

    def on_tick(self):
        # 1) 预览：从服务取最新帧
        frame = self.service.get_preview_frame() if self.service is not None else None
        if frame is not None:
            self.lbl_cam.setPixmap(cv_to_qpixmap(letterbox_square(frame, 480)))

        # 2) 节流触发一次检测 + 分类（使用服务的裁剪保存）
        now = time.time()
        if self.service is None or (now - self._last_detect_t) < self.detect_interval_sec:
            return
        self._last_detect_t = now

        result = self.service.detect_once(save=True)  # 保存 300x300 ROI 到磁盘
        # result 结构: {"ok":bool,"has_hand":bool,"n":int,"saved":[...],"boxes":[...],"confs":[...],"timestamp":str}

        if not result.get("ok", False) or not result.get("has_hand", False) or len(result.get("saved", [])) == 0:
            # 没检测到手
            self.set_topk([])
            self.lbl_crop.setPixmap(QPixmap())
            return

        
        # 3) 读取刚保存的 ROI，做分类与展示
        roi_path = result["saved"][0]  # 若多只手，取第一只；可按需改成最高置信度对应的那张
        roi_img = cv2.imread(roi_path)
        if roi_img is None or roi_img.size == 0:
            self.set_topk([])
            self.lbl_crop.setPixmap(QPixmap())
            return

        # 4) 分类（你的分类器期望 224×224；predict_bgr 内部会 resize 到 size）
        preds = predict_bgr(self.cls_model, self.device, roi_img, size=224, topk=3)
        self.set_topk(preds)
        self.lbl_crop.setPixmap(cv_to_qpixmap(letterbox_square(roi_img, 480)))

def main():
    app = QApplication(sys.argv)
    w = LiveGUI()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
