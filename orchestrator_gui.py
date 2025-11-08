# -*- coding: utf-8 -*-
import sys, os, time
from pathlib import Path
import cv2
import numpy as np
from calibration_api import capture_once, calibrate_all, SAVE_DIR
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from PyQt5.QtWidgets import QSizePolicy
import csv
from datetime import datetime

import mediapipe as mp
import joblib
import tempfile
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

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

from mp_runtime import (
    process_image_with_mediapipe,
    normalize_hand_orientation,
    FeatureCombiner,
    plot_hand_3d_fixed,
)

FEATURE_ORDER = [
    "geo_angle_index_j1","geo_angle_index_j2","geo_angle_index_j3",
    "geo_angle_middle_j1","geo_angle_middle_j2","geo_angle_middle_j3",
    "geo_angle_pinky_j1","geo_angle_pinky_j2","geo_angle_pinky_j3",
    "geo_angle_ring_j1","geo_angle_ring_j2","geo_angle_ring_j3",
    "geo_angle_thumb_j1","geo_angle_thumb_j2","geo_angle_thumb_j3",
    "geo_palm_area",
    "geo_spread_idx_mid","geo_spread_mid_ring","geo_spread_ring_pink",
    "geo_z_max","geo_z_mean","geo_z_min","geo_z_rng","geo_z_std",
    "img_edge_density","img_lap_var"
]

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

        central = QWidget(self)
        self.v = QVBoxLayout(central)
        self.setCentralWidget(central)
        
        # 路径输入
        self.ed_yolo_w = QLineEdit()
        self.ed_cls_w  = QLineEdit()
        btn_yolo_w = QPushButton("Select YOLO Weight(.pt)"); btn_yolo_w.clicked.connect(lambda: self.pick(self.ed_yolo_w, "PyTorch (*.pt)"))
        btn_cls_w  = QPushButton("Select Model Weight(.pt)");   btn_cls_w.clicked.connect(lambda: self.pick(self.ed_cls_w,  "PyTorch (*.pt)"))

        self.btn_start = QPushButton("Active Camera"); self.btn_start.clicked.connect(self.on_start)
        self.btn_stop  = QPushButton("Stop");       self.btn_stop.clicked.connect(self.on_stop); self.btn_stop.setEnabled(False)

        top = QGridLayout()
        top.addWidget(QLabel("YOLO Weight:"),0,0); top.addWidget(self.ed_yolo_w,0,1); top.addWidget(btn_yolo_w,0,2)
        top.addWidget(QLabel("Model:"),1,0);  top.addWidget(self.ed_cls_w,1,1);  top.addWidget(btn_cls_w,1,2)
        top.addWidget(self.btn_start,2,0); top.addWidget(self.btn_stop,2,2)
        
        # === 新增：ROI 保存目录 & Save 按钮 ===
        self.ed_save_dir  = QLineEdit()
        self.btn_pick_dir = QPushButton("Select Save Folder")
        self.btn_pick_dir.clicked.connect(self.pick_dir)

        self.btn_save_roi = QPushButton("💾 Save ROI Now")
        self.btn_save_roi.clicked.connect(self.on_save_roi)

        # 放到 top 布局第 4 行
        top.addWidget(QLabel("Save Folder:"), 4, 0)
        top.addWidget(self.ed_save_dir,        4, 1)
        top.addWidget(self.btn_pick_dir,       4, 2)

        # 第 5 行右侧放“保存 ROI”按钮
        top.addWidget(self.btn_save_roi,       5, 2)

        # === 新增：标定相关按钮 ===
        self.btn_cap_once = QPushButton("📸 Capture Chessboard Image")
        self.btn_calib    = QPushButton("🧮 Run Calibration")
        self.btn_cap_once.clicked.connect(self.on_capture_once)
        self.btn_calib.clicked.connect(self.on_run_calibration)

        # 放在第 3 行两侧
        top.addWidget(self.btn_cap_once, 3, 0)
        top.addWidget(self.btn_calib,    3, 2)

        # === Left live preview ===
        self.lbl_cam = QLabel("Camera")
        self.lbl_cam.setAlignment(Qt.AlignCenter)
        self.lbl_cam.setMinimumSize(560, 420)
        self.lbl_cam.setStyleSheet("background:#111;color:#aaa;")

        # === Right: 2×2 grid panels ===
        def _make_panel(title: str) -> QLabel:
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(360, 270)
            lbl.setStyleSheet("background:#111;color:#aaa;")
            return lbl

        self.lbl_calib = _make_panel("Calibration Preview")  # 左上
        
        self.lbl_roi = QLabel("Recognition ROI")
        self.lbl_roi.setAlignment(Qt.AlignCenter)
        self.lbl_roi.setFixedSize(300, 300)
        self.lbl_roi.setScaledContents(False)
        self.lbl_roi.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_roi.setStyleSheet("background:#111;color:#aaa;")
        
        self.lbl_aux1  = _make_panel("Spare 1")  # 左下
        self.lbl_aux2  = _make_panel("Spare 2")  # 右下

        right_grid = QGridLayout()
        right_grid.setContentsMargins(0, 0, 0, 0)
        right_grid.setSpacing(6)
        right_grid.addWidget(self.lbl_calib, 0, 0)
        right_grid.addWidget(self.lbl_roi,   0, 1)
        right_grid.addWidget(self.lbl_aux1,  1, 0)
        right_grid.addWidget(self.lbl_aux2,  1, 1)

        right_panel = QWidget()
        right_panel.setLayout(right_grid)

        # === Combine left & right ===
        imgs = QHBoxLayout()
        imgs.addWidget(self.lbl_cam, 1)
        imgs.addWidget(right_panel, 1)

        # === 统一的预测行（CNN / SVM / KNN） ===
        def _pred_card(title, color):
            box = QGroupBox(title)
            box.setStyleSheet("QGroupBox { font-weight: 600; }")
            v = QVBoxLayout(box)
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(36)
            lbl.setStyleSheet(
                f"QLabel {{ "
                f"  font-size: 18px; font-weight: 600; "
                f"  color: {color}; "
                f"  padding: 2px 6px; "
                f"  border: 1px solid #444; border-radius: 6px; "
                f"  background: #111;"
                f"}}"
            )
            v.addWidget(lbl)
            return box, lbl

        # 仅给 CNN 用：三行 Top-3
        def _pred_card3(title):
            box = QGroupBox(title)
            box.setStyleSheet("QGroupBox { font-weight: 600; }")
            v = QVBoxLayout(box)
            lbls = []
            sizes = [18, 15, 14]
            colors = ["#ffd54f", "#c5e1a5", "#81d4fa"]
            for i in range(3):
                lbl = QLabel("—")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setMinimumHeight(28)
                lbl.setStyleSheet(
                    f"QLabel {{ "
                    f"  font-size: {sizes[i]}px; font-weight: 600; "
                    f"  color: {colors[i]}; "
                    f"  padding: 2px 6px; "
                    f"  border: 1px solid #444; border-radius: 6px; "
                    f"  background: #111;"
                    f"}}"
                )
                v.addWidget(lbl)
                lbls.append(lbl)
            return box, lbls

        self.pred_row = QHBoxLayout()
        self.pred_row.setSpacing(12)

        # 把原来的 CNN 单行卡片替换为三行 Top-3：
        card_cnn,  cnn_lbls  = _pred_card3("ResNet (CNN)")
        self.lbl_cnn_top1, self.lbl_cnn_top2, self.lbl_cnn_top3 = cnn_lbls

        # SVM / KNN 保持单行
        card_svm,  self.lbl_svm_pred  = _pred_card("SVM",          "#ff5252")
        card_knn,  self.lbl_knn_pred  = _pred_card("KNN",          "#40c4ff")

        self.pred_row.addWidget(card_cnn, 1)
        self.pred_row.addWidget(card_svm, 1)
        self.pred_row.addWidget(card_knn, 1)

        # === 布局挂载 ===
        self.v.addLayout(top)
        self.v.addSpacing(6)
        self.v.addLayout(imgs)
        self.v.addSpacing(8)
        self.v.addLayout(self.pred_row)
        self.v.addSpacing(6)

        # === 标定状态标签（再往下） ===
        self.lbl_calib_status = QLabel("Calibration: —")
        self.lbl_calib_status.setStyleSheet("color:#000;")
        self.v.addWidget(self.lbl_calib_status)

        # ===== 录制控件（连续 N 次预测事件）=====
        self.ed_nframes = QLineEdit("15")  # 默认 30 次（改）
        self.ed_nframes.setFixedWidth(80)
        self.btn_record_start = QPushButton("Start Recording")
        self.btn_record_stop  = QPushButton("Stop Recording")
        self.btn_record_stop.setEnabled(False)
        self.btn_record_start.clicked.connect(self.on_record_start)
        self.btn_record_stop.clicked.connect(self.on_record_stop)

        rec_row = QGridLayout()
        rec_row.addWidget(QLabel("Frames:"), 0, 0)
        rec_row.addWidget(self.ed_nframes,   0, 1)
        rec_row.addWidget(self.btn_record_start, 0, 2)
        rec_row.addWidget(self.btn_record_stop,  0, 3)
        self.v.addLayout(rec_row)

        self.lbl_rec_status = QLabel("Recorder: idle")
        self.v.addWidget(self.lbl_rec_status)

        # —— 运行时：MediaPipe Hands & SVM 模型（只初始化一次）——
        self._mp_hands = None
        self._svm_model = None
        self._knn_model = None 

        # 状态
        self.timer = QTimer(self); self.timer.timeout.connect(self.on_tick)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ▼ 服务实例与节流
        self.service = None
        self._last_detect_t = 0.0
        self.detect_interval_sec = 0.35  # 每隔 ~350ms 触发一次 detect_once()

        # 分类器
        self.cls_model = None
        # detection cache (for overlay + sticky ROI)
        self.last_boxes = []         # list of [x1,y1,x2,y2]
        self.last_confs = []         # list of confidences (optional)
        self.last_roi_img = None     # last successful ROI (numpy image)

        # ===== Recorder state（仅保存三个标签 & 宽表CSV） =====
        self._rec_active: bool = False
        self._rec_buffer: list = []   # 存放每次预测的 [cnn, svm, knn]
        self._rec_target: int = 30    # 默认 30（改）
        self._rec_count: int = 0
        self._rec_out_path: str = None
        self._prev_center = None  # for Δcenter calculation

    # ---------------- Recorder helpers ----------------
    def _record_folder(self) -> str:
        # Prefer user's Save Folder; else fallback to ./records/runs
        base = self.ed_save_dir.text().strip()
        if not base:
            base = os.path.join(os.getcwd(), "records")
        out = os.path.join(base, "runs")
        os.makedirs(out, exist_ok=True)
        return out

    def _new_record_path(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"record_{ts}_{self._rec_target}f.csv"
        return os.path.join(self._record_folder(), fname)

    # —— 新增：宽表CSV路径
    def _matrix_csv_path(self) -> str:
        return os.path.join(self._record_folder(), "angle_s2.csv")

    # —— 新增：把 self._rec_buffer 作为新一轮列追加到宽表 CSV
    def _append_matrix_columns(self):
        """
        self._rec_buffer: [[cnn, svm, knn], ...]  共 N 行
        pred_matrix.csv: 追加列 CNN_Rk, SVM_Rk, KNN_Rk
        """
        buf = self._rec_buffer
        path = self._matrix_csv_path()

        # 读旧文件
        old_header = []
        old_rows = []
        if os.path.exists(path):
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        old_header = row
                    else:
                        old_rows.append(row)

        # 下一轮编号 k
        k = 0
        for name in old_header:
            if name.startswith("CNN_R"):
                try:
                    k = max(k, int(name.split("_R")[-1]))
                except:
                    pass
        k += 1

        new_cols = [f"CNN_R{k}", f"SVM_R{k}", f"KNN_R{k}"]

        # 行数对齐
        max_rows = max(len(old_rows), len(buf))
        # 旧表行宽
        old_width = len(old_header)

        # 补齐旧表行/列
        while len(old_rows) < max_rows:
            old_rows.append([])
        for r in old_rows:
            while len(r) < old_width:
                r.append("")

        # 生成新列数据
        for i in range(max_rows):
            if i < len(buf):
                cn, sv, kn = buf[i]
            else:
                cn, sv, kn = "", "", ""
            if i < len(old_rows):
                old_rows[i].extend([cn, sv, kn])
            else:
                row = ([""] * old_width) + [cn, sv, kn]
                old_rows.append(row)

        # 写回
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(old_header + new_cols if old_header else new_cols)
            for r in old_rows:
                writer.writerow(r)

        QMessageBox.information(self, "Recording",
            f"Saved {len(self._rec_buffer)} predictions to:\n{path}")

    def on_record_start(self):
        try:
            n = int(self.ed_nframes.text())
            if n <= 0: raise ValueError
        except Exception:
            QMessageBox.information(self, "Recording", "Frames must be a positive integer.")
            return
        self._rec_active = True
        self._rec_buffer = []
        self._rec_target = n
        self._rec_count  = 0
        self._rec_out_path = None  # 改：不再使用逐行CSV
        self._prev_center = None
        self.btn_record_start.setEnabled(False)
        self.btn_record_stop.setEnabled(True)
        self.lbl_rec_status.setText(f"Recorder: capturing {n} prediction events …")
        self.statusBar().showMessage("Recording → pred_matrix.csv (wide table)")

    def on_record_stop(self):
        if not self._rec_active:
            return
        self._rec_active = False
        # 改：写入宽表CSV
        self._append_matrix_columns()
        self.btn_record_start.setEnabled(True)
        self.btn_record_stop.setEnabled(False)
        self.lbl_rec_status.setText("Recorder: idle")
        self.statusBar().showMessage("Recording stopped")

    def _flush_record_buffer(self):
        # 保留原函数以最小改动；此版本不再使用逐行CSV，留空
        return

    # ---------------- File pickers ----------------
    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if d:
            self.ed_save_dir.setText(d)

    def on_save_roi(self):
        if self.last_roi_img is None or self.last_roi_img.size == 0:
            QMessageBox.information(self, "Save ROI", "No ROI to save yet.")
            return
        save_dir = self.ed_save_dir.text().strip()
        if not save_dir:
            QMessageBox.information(self, "Save ROI", "Please select a save folder first.")
            return
        os.makedirs(save_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        out_path = os.path.join(save_dir, f"roi_{ts}_{ms:03d}.jpg")
        _ = cv2.imwrite(out_path, self.last_roi_img)

    def pick(self, line: QLineEdit, filt: str):
        p, _ = QFileDialog.getOpenFileName(self, "Select File", "", filt)
        if p: line.setText(p)

    # ---------------- Start/Stop camera ----------------
    def on_start(self):
        if not Path(self.ed_yolo_w.text()).exists() or not Path(self.ed_cls_w.text()).exists():
            QMessageBox.information(self,"提示","Select YOLO and model .pt"); return

        if self.cls_model is None:
            self.statusBar().showMessage(f"Loading classifier on {self.device} …")
            self.cls_model = load_model(self.ed_cls_w.text(), self.device, num_classes=len(CLASSES))

        if self.service is None:
            self.statusBar().showMessage("Starting camera service …")
            self.service = HandCamService(
                model_path=self.ed_yolo_w.text(),
                cam_index = 0,
                imgsz = 640,
                conf_thr = 0.28,
                iou_thr = 0.55,
                target_size = (300, 300),  # 最终输出尺寸
                pad_color = (255, 255, 255),  # 白边
                debug_draw = False,
            )
            self.service.start()
            self._last_detect_t = 0.0

        # 模型只加载一次
        if self._svm_model is None:
            svm_pkl_path = r"D:\Files\2025_Y4_S2\AMME5710\Major\svm_model.pkl"
            if not Path(svm_pkl_path).exists():
                QMessageBox.information(self, "提示", f"SVM 模型不存在：\n{svm_pkl_path}")
                return
            self._svm_model = joblib.load(svm_pkl_path)
            self.statusBar().showMessage("SVM loaded.")

        if self._knn_model is None:
            knn_pkl_path = r"D:\Files\2025_Y4_S2\AMME5710\Major\knn_model.pkl"
            if not Path(knn_pkl_path).exists():
                QMessageBox.information(self, "提示", f"KNN 模型不存在：\n{knn_pkl_path}")
                return
            self._knn_model = joblib.load(knn_pkl_path)
            self.statusBar().showMessage("KNN loaded.")

        if self._mp_hands is None:
            mp_hands = mp.solutions.hands
            self._mp_hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.80,
                model_complexity=1
            )

        # 3) 开计时器（约 20 fps）
        self.timer.start(50)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Running …")

    def on_stop(self):
        # 若正在录制，先安全收尾
        if self._rec_active:
            self.on_record_stop()

        self.timer.stop()
        if self.service is not None:
            self.service.stop()
            self.service = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    # ---------------- Calibration helpers ----------------
    def on_capture_once(self):
        self.statusBar().showMessage("Capturing chessboard image…")

        frame = None
        if self.service is not None:
            frame = self.service.get_preview_frame()

        if frame is None:
            try:
                res = capture_once(camera_index=0)
            except Exception as e:
                self.statusBar().showMessage(f"Capture error: {e}")
                QMessageBox.warning(self, "Capture", f"Capture failed: {e}")
                return

            if not res.get("ok", False):
                msg = res.get("msg", "Capture failed")
                self.statusBar().showMessage(msg)
                QMessageBox.information(self, "Capture", msg)
                self.lbl_calib_status.setText("Calibration: Capture failed")
                return

            img_path = res.get("image_path")
            vis_path = res.get("corners_preview_path")
            detected = res.get("chessboard_detected", False)
        else:
            os.makedirs(SAVE_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(SAVE_DIR, f"img_{ts}.jpg")
            cv2.imwrite(img_path, frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                ret, corners = cv2.findChessboardCornersSB(
                    gray, (9, 6), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if not ret:
                    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    ret, corners = cv2.findChessboardCorners(gray, (9, 6), flags)
                    if ret:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), flags)
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            detected = bool(ret)
            vis_path = None
            if detected:
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, (9, 6), corners.astype(np.float32), True)
                vis_path = img_path.replace(".jpg", "_corners.jpg")
                cv2.imwrite(vis_path, vis)

        show_path = vis_path if (detected and vis_path and os.path.exists(vis_path)) else img_path
        if show_path and os.path.exists(show_path):
            img = cv2.imread(show_path)
            if img is not None and img.size > 0:
                self.lbl_calib.setPixmap(cv_to_qpixmap(letterbox_square(img, 480)))

        msg = f"Saved: {os.path.basename(img_path)} | chessboard={'YES' if detected else 'NO'}"
        self.statusBar().showMessage(msg)
        self.lbl_calib_status.setText(f"Calibration: {msg}")
        QMessageBox.information(self, "Capture", msg)

    def on_run_calibration(self):
        self.statusBar().showMessage("Running calibration… (this may take a few seconds)")
        QApplication.processEvents()

        try:
            res = calibrate_all()
        except Exception as e:
            self.statusBar().showMessage(f"Calibration error: {e}")
            QMessageBox.critical(self, "Calibration", f"Calibrition failed：{e}")
            return

        if not res.get("ok", False):
            ok_cnt = sum(1 for r in res.get("per_image_report",[]) if r.get("ok"))
            total  = len(res.get("per_image_report",[]))
            detail = res.get("msg","Calibration failed")
            self.lbl_calib_status.setText(f"Calibration: {detail} ({ok_cnt}/{total} images ok)")
            self.statusBar().showMessage(detail)
            QMessageBox.warning(self, "Calibration", f"{detail}\nValid images: {ok_cnt}/{total}\nTry capturing more chessboard images from different angles.")
            return

        rms   = res.get("rms")
        meanE = res.get("mean_reprojection_error")
        K     = res.get("camera_matrix")
        dist  = res.get("dist_coeffs")
        imsz  = res.get("image_size",{})

        summary = (
            f"Calibration successful!\n"
            f"- Image size: {imsz.get('width')}x{imsz.get('height')}\n"
            f"- RMS: {rms:.4f}\n"
            f"- Mean reprojection error: {meanE:.4f}\n"
            f"- K (camera matrix):\n  {np.array(K)}\n"
            f"- Dist coeffs:\n  {np.array(dist)}\n"
            f"- Saved to calibration_result.json"
        )

        self.lbl_calib_status.setText(f"Calibration: OK | RMS={rms:.4f}, MeanErr={meanE:.4f}")
        self.statusBar().showMessage("Calibration done.")
        QMessageBox.information(self, "Calibration", summary)

    # ---------------- Draw / overlay ----------------
    def _draw_boxes(self, img_bgr, boxes, confs=None):
        if img_bgr is None or not boxes:
            return
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if confs is not None and i < len(confs):
                txt = f"{confs[i]:.2f}"
                cv2.putText(img_bgr, txt, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    def show_mediapipe_overlay(self, img_bgr, panel_size=300):
        if img_bgr is None or img_bgr.size == 0 or self._mp_hands is None:
            self.lbl_aux1.setPixmap(QPixmap())
            self.lbl_aux1.setText("Spare 1")
            return

        try:
            r = process_image_with_mediapipe(img_bgr, self._mp_hands)
        except Exception as e:
            self.statusBar().showMessage(f"mediapipe overlay error: {e}")
            self.lbl_aux1.setPixmap(QPixmap())
            self.lbl_aux1.setText("Spare 1")
            return

        if not r or r.get("n_hands", 0) == 0:
            self.lbl_aux1.setPixmap(QPixmap())
            self.lbl_aux1.setText("No hand")
            return

        hinfo = r["hands"][0]
        LM = hinfo["landmarks_px_xyz"]
        overlay = img_bgr.copy()

        h, w = overlay.shape[:2]
        nlms = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(
                    x=float(max(0.0, min(1.0, LM[i,0] / max(1, w)))),
                    y=float(max(0.0, min(1.0, LM[i,1] / max(1, h)))),
                    z=float(LM[i,2])
                ) for i in range(21)
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            overlay,
            nlms,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        tile = letterbox_square(overlay, size=panel_size, color=(17,17,17))
        self.lbl_aux1.setPixmap(cv_to_qpixmap(tile))

    # ---------------- Main tick ----------------
    def on_tick(self):
        # 1) 预览：从服务取最新帧
        frame = self.service.get_preview_frame() if self.service is not None else None
        if frame is not None:
            disp = frame.copy()
            self._draw_boxes(disp, self.last_boxes, self.last_confs)
            self.lbl_cam.setPixmap(cv_to_qpixmap(letterbox_square(disp, 480)))

        # 2) 节流触发一次检测 + 分类（直接使用 YOLO ROI）
        now = time.time()
        if self.service is None or (now - self._last_detect_t) < self.detect_interval_sec:
            return
        self._last_detect_t = now

        result = self.service.detect_once(save=True)  # YOLO服务保存ROI
        # 期望结构: {"ok":bool,"has_hand":bool,"n":int,"saved":[...],"boxes":[...],"confs":[...],"timestamp":str}
        if not result.get("ok", False):
            self.last_boxes, self.last_confs = [], []
            return

        if not result.get("has_hand", False) or len(result.get("saved", [])) == 0:
            self.last_boxes, self.last_confs = [], []
            # 清空显示
            self.lbl_cnn_top1.setText("—")
            self.lbl_cnn_top2.setText("—")
            self.lbl_cnn_top3.setText("—")
            self.lbl_svm_pred.setText("—")
            self.lbl_knn_pred.setText("—")
            return

        # --- 检测成功：更新框 + ROI ---
        self.last_boxes = result.get("boxes", []) or []
        self.last_confs = result.get("confs", []) or []

        roi_path = result["saved"][0]
        roi_img = cv2.imread(roi_path)
        if roi_img is not None and roi_img.size > 0:
            self.last_roi_img = roi_img
            self.lbl_roi.setText("")
            self.lbl_roi.setPixmap(cv_to_qpixmap(roi_img))
            self.show_mediapipe_overlay(self.last_roi_img, panel_size=300)
        else:
            # 无法读取则维持上一次 ROI
            pass

        # --- CNN 分类（显示 Top-3） ---
        preds = []  # 确保后续可用
        try:
            roi_224 = letterbox_square(roi_img, size=224, color=(255, 255, 255))
            preds = predict_bgr(self.cls_model, self.device, roi_224, size=224, topk=3)

            self.lbl_cnn_top1.setText("—")
            self.lbl_cnn_top2.setText("—")
            self.lbl_cnn_top3.setText("—")

            if preds and len(preds) > 0:
                def _fmt(tup):
                    cls_name, prob = tup
                    return f"{cls_name}  ({prob:.2f})"
                if len(preds) >= 1:
                    self.lbl_cnn_top1.setText(_fmt(preds[0]))
                if len(preds) >= 2:
                    self.lbl_cnn_top2.setText(_fmt(preds[1]))
                if len(preds) >= 3:
                    self.lbl_cnn_top3.setText(_fmt(preds[2]))
        except Exception as e:
            self.statusBar().showMessage(f"CNN classify error: {e}")
            self.lbl_cnn_top1.setText("—")
            self.lbl_cnn_top2.setText("—")
            self.lbl_cnn_top3.setText("—")

        # ===== 基于 ROI 做 MediaPipe → Lc → SVM/KNN 预测 =====
        if (self._mp_hands is not None) and (self._svm_model is not None) and (self.last_roi_img is not None):
            try:
                roi_img = self.last_roi_img
                h, w = roi_img.shape[:2]

                r = process_image_with_mediapipe(roi_img, self._mp_hands)
                if r and r.get("n_hands", 0) > 0:
                    hinfo = r["hands"][0]
                    feat63 = hinfo["feature63"].astype("float32")  # (63,)
                    L = feat63.reshape((21, 3))

                    Lc, info = normalize_hand_orientation(L, method="basis",
                                                          kabsch_with_scale=True,
                                                          mirror_thumb=True)
                    Lc[:, 0] *= 2.5

                    # 可视化 3D（右下）
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        save_png = tf.name
                    plot_hand_3d_fixed(Lc, assume_normalized=True, elev=20, azim=120, save_path=save_png)
                    img3d = cv2.imread(save_png)
                    if img3d is not None and img3d.size > 0:
                        img3d_small = cv2.resize(img3d, (300, 300), interpolation=cv2.INTER_AREA)
                        self.lbl_aux2.setPixmap(cv_to_qpixmap(img3d_small))
                    try:
                        os.remove(save_png)
                    except Exception:
                        pass

                    combiner = FeatureCombiner(include_original63=False)
                    feats = combiner.compute(roi_img, Lc)
                    x = np.array([feats.get(k, np.nan) for k in FEATURE_ORDER],
                                 dtype=np.float32).reshape(1, -1)

                    svm_text = "—"
                    knn_text = "—"
                    try:
                        if self._svm_model is not None:
                            y_pred_svm = self._svm_model.predict(x)[0]
                            svm_text = f"{y_pred_svm}"
                    except Exception as e:
                        self.statusBar().showMessage(f"SVM predict error: {e}")

                    try:
                        if self._knn_model is not None:
                            y_pred_knn = self._knn_model.predict(x)[0]
                            knn_text = f"{y_pred_knn}"
                    except Exception as e:
                        self.statusBar().showMessage(f"KNN predict error: {e}")

                    self.lbl_svm_pred.setText(svm_text)
                    self.lbl_knn_pred.setText(knn_text)

                else:
                    self.lbl_cnn_top1.setText("—")
                    self.lbl_cnn_top2.setText("—")
                    self.lbl_cnn_top3.setText("—")
                    self.lbl_svm_pred.setText("—")
                    self.lbl_knn_pred.setText("—")

            except Exception as e:
                self.statusBar().showMessage(f"MediaPipe/SVM error: {e}")

        # ===== 录制：只存 CNN/SVM/KNN 标签，并在达标后追加列到宽表 =====
        if self._rec_active:
            # 取 CNN 标签（top1 文本形如 "A  (0.97)"）
            cnn_label = ""
            t = self.lbl_cnn_top1.text().strip()
            if t not in ("—", ""):
                cnn_label = t.split()[0]

            svm_label = self.lbl_svm_pred.text().strip()
            knn_label = self.lbl_knn_pred.text().strip()

            prediction_event = any(v not in ("—", "", None) for v in (cnn_label, svm_label, knn_label))
            if prediction_event:
                self._rec_buffer.append([cnn_label, svm_label, knn_label])
                self._rec_count += 1
                self.lbl_rec_status.setText(f"Recorder: {self._rec_count}/{self._rec_target}")

                if self._rec_count >= self._rec_target:
                    self._rec_active = False
                    self._append_matrix_columns()  # 改：写宽表CSV
                    self.btn_record_start.setEnabled(True)
                    self.btn_record_stop.setEnabled(False)
                    self.lbl_rec_status.setText("Recorder: idle")

def main():
    app = QApplication(sys.argv)
    w = LiveGUI()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
