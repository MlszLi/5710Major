# # -*- coding: utf-8 -*-
# """
# hand_cam_service.py
# --------------------------------
# 常开摄像头 (后台线程更新最新帧)；仅在外部调用 detect_once() 时执行 YOLO 检测并可选保存裁剪 ROI。
# - GUI 可持续显示 preview (get_preview_frame())
# - 当用户点按钮/定时触发时调用 detect_once()
# - 检测到手则保存 300x300 的等比例+白边填充的纯净 ROI（无边框），并返回路径；未检测到手返回空列表。

# 依赖：
#   pip install ultralytics opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# """

# import os
# import cv2
# import time
# import threading
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple
# from ultralytics import YOLO


# # -------------------- 工具函数 --------------------

# def ensure_dir(p: str):
#     if not os.path.exists(p):
#         os.makedirs(p, exist_ok=True)


# def timestamp() -> str:
#     return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# def expand_box(xyxy, W, H, r=0.25):
#     """在原 bbox 基础上按比例向外扩展"""
#     x1, y1, x2, y2 = map(int, xyxy)
#     w, h = x2 - x1, y2 - y1
#     dx, dy = int(w * r), int(h * r)
#     x1 = max(0, x1 - dx)
#     y1 = max(0, y1 - dy)
#     x2 = min(W, x2 + dx)
#     y2 = min(H, y2 + dy)
#     return x1, y1, x2, y2


# def resize_with_padding(img, target_size: Tuple[int, int] = (300, 300), pad_color=(255, 255, 255)):
#     """
#     等比例缩放到不超过 target_size，再用指定颜色填充到精确的 target_size。
#     这样既不拉伸形变，也不会截掉“数字1”等细长手势。
#     """
#     th, tw = target_size[1], target_size[0]  # 注意 OpenCV 的 (w,h) vs (h,w)
#     h, w = img.shape[:2]
#     # 计算等比例缩放
#     scale = min(tw / w, th / h)
#     new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     # 计算四周填充
#     pad_w = (tw - new_w) // 2
#     pad_h = (th - new_h) // 2
#     result = cv2.copyMakeBorder(
#         resized,
#         pad_h, th - new_h - pad_h,
#         pad_w, tw - new_w - pad_w,
#         cv2.BORDER_CONSTANT, value=pad_color
#     )
#     return result


# # -------------------- 主服务类 --------------------

# class HandCamService:
#     def __init__(
#         self,
#         model_path: str,
#         root_dir: str = r"D:\Files\2025_Y4_S2\AMME5710\Major",
#         save_root: Optional[str] = None,
#         cam_index: int = 0,
#         imgsz: int = 640,
#         conf_thr: float = 0.28,
#         iou_thr: float = 0.55,
#         expand_r: float = 0.2,          # 扩框默认更大，避免手指被截
#         target_size: Tuple[int, int] = (300, 300),  # 最终输出尺寸
#         pad_color: Tuple[int, int, int] = (255, 255, 255),  # 白边
#         preview_width: int = 1280,
#         preview_height: int = 720,
#     ):
#         self.model_path = model_path
#         self.root_dir = root_dir
#         # 单独放 300x300 的输出，避免和旧 256x256 混在一起
#         self.save_root = save_root or os.path.join(root_dir, r"Root\Video")
#         self.cam_index = cam_index
#         self.imgsz = imgsz
#         self.conf_thr = conf_thr
#         self.iou_thr = iou_thr
#         self.expand_r = expand_r
#         self.target_size = target_size
#         self.pad_color = pad_color
#         self.preview_w = preview_width
#         self.preview_h = preview_height

#         ensure_dir(self.save_root)
#         self.day_dir = os.path.join(self.save_root, datetime.now().strftime("%Y%m%d"))
#         ensure_dir(self.day_dir)

#         # 运行态
#         self._cap = None
#         self._reader_thread = None
#         self._stop_flag = threading.Event()
#         self._lock = threading.Lock()
#         self._latest_frame = None  # 最新原始帧 (BGR)
#         self._model = None         # 懒加载

#     # --------------- 生命周期 ---------------

#     def start(self):
#         """启动摄像头与取帧线程；加载YOLO模型"""
#         if self._cap is not None:
#             return
#         print(f"[INFO] Loading model: {self.model_path}")
#         self._model = YOLO(self.model_path)

#         cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
#         if not cap.isOpened():
#             print("[WARN] CAP_DSHOW 打不开，尝试默认后端...")
#             cap = cv2.VideoCapture(self.cam_index)
#         if not cap.isOpened():
#             raise RuntimeError("无法打开摄像头，请检查 CAM_INDEX 或权限。")

#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_w)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_h)

#         self._cap = cap
#         self._stop_flag.clear()
#         self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
#         self._reader_thread.start()
#         print("[INFO] HandCamService started.")

#     def stop(self):
#         """停止线程并释放摄像头资源"""
#         self._stop_flag.set()
#         if self._reader_thread is not None:
#             self._reader_thread.join(timeout=1.0)
#             self._reader_thread = None
#         if self._cap is not None:
#             self._cap.release()
#             self._cap = None
#         print("[INFO] HandCamService stopped.")

#     def _reader_loop(self):
#         """后台线程：持续读取最新帧"""
#         while not self._stop_flag.is_set():
#             ok, frame = self._cap.read()
#             if not ok:
#                 time.sleep(0.005)
#                 continue
#             with self._lock:
#                 self._latest_frame = frame  # 原始BGR

#     # --------------- 外部接口 ---------------

#     def get_preview_frame(self):
#         """返回最新帧的拷贝（BGR）；无帧则返回 None。GUI 可用它来显示实时画面。"""
#         with self._lock:
#             if self._latest_frame is None:
#                 return None
#             return self._latest_frame.copy()

#     def detect_once(self, save: bool = True) -> Dict[str, Any]:
#         """
#         执行一次YOLO检测（使用当前最新帧），按需保存 300x300（等比例+白边）的 ROI 并返回结果。
#         返回字典示例：
#           {
#             "ok": True/False,
#             "has_hand": True/False,
#             "n": 2,
#             "saved": [".../20251028/20251028_120000_123456_0.png", ...],
#             "boxes": [[x1,y1,x2,y2], ...],   # 原始bbox（未扩展）
#             "confs": [0.91, ...],
#             "timestamp": "YYYYmmdd_HHMMSS_xxxxxx"
#           }
#         """
#         # 取最新原始帧（干净帧）
#         with self._lock:
#             if self._latest_frame is None:
#                 return {"ok": False, "has_hand": False, "n": 0, "saved": [], "boxes": [], "confs": []}
#             raw = self._latest_frame.copy()

#         H, W = raw.shape[:2]
#         # 推理
#         res = self._model.predict(
#             source=raw,
#             imgsz=self.imgsz,
#             conf=self.conf_thr,
#             iou=self.iou_thr,
#             verbose=False
#         )[0]
#         boxes_obj = res.boxes
#         if len(boxes_obj) == 0:
#             return {"ok": True, "has_hand": False, "n": 0, "saved": [], "boxes": [], "confs": [], "timestamp": timestamp()}

#         # 解析结果
#         xyxy_list, confs = [], []
#         for b in boxes_obj:
#             x1, y1, x2, y2 = map(int, b.xyxy[0])
#             xyxy_list.append([x1, y1, x2, y2])
#             confs.append(float(b.conf[0]))

#         saved_paths: List[str] = []
#         ts = timestamp()
#         if save:
#             sub_dir = os.path.join(self.day_dir, ts[:8])
#             ensure_dir(sub_dir)
#             for i, (x1, y1, x2, y2) in enumerate(xyxy_list):
#                 # 先扩框，保证细长手势不被截
#                 ex1, ey1, ex2, ey2 = expand_box([x1, y1, x2, y2], W, H, self.expand_r)
#                 crop = raw[ey1:ey2, ex1:ex2]
#                 if crop.size == 0:
#                     continue
#                 # 再做等比例缩放 + 白边填充到 300x300
#                 crop_square = resize_with_padding(crop, self.target_size, self.pad_color)
#                 out_path = os.path.join(sub_dir, f"{ts}_{i}.png")
#                 cv2.imwrite(out_path, crop_square)
#                 saved_paths.append(out_path)

#         return {
#             "ok": True,
#             "has_hand": True,
#             "n": len(xyxy_list),
#             "saved": saved_paths,
#             "boxes": xyxy_list,   # 注意：这里返回的是原始 bbox（未扩展），如需扩展后的也可返回
#             "confs": confs,
#             "timestamp": ts
#         }


# # --------------------- 简易验证主循环（示例） ---------------------
# # GUI 程序里可以不用下面的 main；这里演示“隔几秒调用 detect_once()”。

# if __name__ == "__main__":
#     MODEL = r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\yolo\best.pt"  # 用你最新的 best.pt
#     service = HandCamService(
#         model_path=MODEL,
#         root_dir=r"D:\Files\2025_Y4_S2\AMME5710\Major",
#         cam_index=0,
#         imgsz=640,
#         conf_thr=0.28,
#         iou_thr=0.55,
#         expand_r=0.3,          # 截图范围更大
#         target_size=(300, 300), # 输出 300x300
#         pad_color=(255, 255, 255)
#     )

#     service.start()
#     print("[INFO] Press 'SPACE' to trigger detect_once(); 'ESC' to quit.")

#     last_call = 0.0
#     interval = 3.0  # 每隔3秒调用一次（模拟 GUI 定时）
#     try:
#         while True:
#             frame = service.get_preview_frame()
#             if frame is not None:
#                 cv2.imshow("Preview (no YOLO unless called)", frame)

#             # 定时触发一次检测
#             now = time.time()
#             if now - last_call >= interval:
#                 result = service.detect_once(save=True)
#                 print("[CALL detect_once]:", result)
#                 last_call = now

#             key = cv2.waitKey(1) & 0xFF
#             if key in (27, ord('q'), ord('Q')):  # ESC/Q 退出
#                 break
#             elif key == 32:  # 空格手动触发一次
#                 result = service.detect_once(save=True)
#                 print("[SPACE detect_once]:", result)

#     finally:
#         service.stop()
#         cv2.destroyAllWindows()






# -*- coding: utf-8 -*-
"""
hand_cam_service.py
--------------------------------
常开摄像头 (后台线程更新最新帧)；仅在外部调用 detect_once() 时执行 YOLO 检测并可选保存裁剪 ROI。
- GUI 可持续显示 preview (get_preview_frame())
- 当用户点按钮/定时触发时调用 detect_once()
- 检测到手则保存 300x300 的等比例+白边填充的纯净 ROI（无边框），并返回路径；未检测到手返回空列表。

新增：
1) 预览与裁剪分离：_latest_raw（干净源）用于检测与裁剪；_latest_frame 用于预览（可叠加可视化）。
2) 各向异性自适应裁剪：按手势方向自适应长/短轴留白（竖直 → 左右更紧；水平 → 上下更紧）。
3) 调试可视化：debug_draw=True 时，预览上画原始bbox(绿色)和最终裁剪框(蓝色)。

依赖：
  pip install ultralytics opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

import os
import cv2
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO


# -------------------- 工具函数 --------------------

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resize_with_padding(img, target_size: Tuple[int, int] = (300, 300), pad_color=(255, 255, 255)):
    """
    等比例缩放到不超过 target_size，再用指定颜色填充到精确的 target_size。
    """
    th, tw = target_size[1], target_size[0]  # 注意 OpenCV 的 (w,h) vs (h,w)
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (tw - new_w) // 2
    pad_h = (th - new_h) // 2
    result = cv2.copyMakeBorder(
        resized,
        pad_h, th - new_h - pad_h,
        pad_w, tw - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return result


# -------------------- 各向异性自适应裁剪 --------------------

def clamp_rect_to_image(x1, y1, x2, y2, W, H):
    """优先平移把矩形拉回画面，仍越界再截断。"""
    dx_left  = max(0 - x1, 0)
    dy_top   = max(0 - y1, 0)
    dx_right = min(W - x2, 0)
    dy_bot   = min(H - y2, 0)
    x1 += dx_left + dx_right
    x2 += dx_left + dx_right
    y1 += dy_top  + dy_bot
    y2 += dy_top  + dy_bot
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    return int(x1), int(y1), int(x2), int(y2)


def rect_expand_anisotropic_adaptive(
    xyxy, W, H,
    base_short=0.06,     # 短轴基础留白（越小越紧）
    base_long=0.22,      # 长轴基础留白（越大越松）
    gain_long=0.15,      # 近距离时长轴额外留白
    shrink_short=0.25,   # 近距离时短轴按比例再收一点
    up_base=0.06,        # 竖直手基础向上偏置（相对 bbox 高度）
    up_gain=0.16,        # 近距离竖直手额外向上偏置
    min_side_px=64
):
    """
    各向异性矩形裁剪：
      - 竖直手：上下(长轴) 留白略大，左右(短轴) 紧；再上偏 up_bias
      - 水平手：左右(长轴) 留白略大，上下(短轴) 紧
      - 中性：两轴折中
    最终仍送入 resize_with_padding(...) 生成 300x300。
    """
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        s = max(min_side_px, 32)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return clamp_rect_to_image(int(cx - s/2), int(cy - s/2), int(cx + s/2), int(cy + s/2), W, H)

    # 手的相对大小（0~1），大表示更近
    size_rel = max(h / H, w / W)
    t = (size_rel - 0.25) / (0.75 - 0.25)
    t = 0.0 if t < 0 else (1.0 if t > 1.0 else t)

    aspect = h / max(w, 1e-6)
    is_vert  = aspect >= 1.15
    is_horiz = aspect <= 0.87

    if is_vert:
        # 竖直：长轴=height，短轴=width
        m_long  = base_long  + gain_long * t
        m_short = base_short * (1.0 - shrink_short * t)
        up_bias = up_base + up_gain * t + (0.06 if (y1 / H) < 0.12 else 0.0)  # 顶部很近再加点
        dx = w * m_short
        dy = h * m_long
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0 - up_bias * h
        rx1, ry1 = cx - (w/2 + dx), cy - (h/2 + dy)
        rx2, ry2 = cx + (w/2 + dx), cy + (h/2 + dy)

    elif is_horiz:
        # 水平：长轴=width，短轴=height（上下要紧）
        m_long  = base_long  + gain_long * t
        m_short = base_short * (1.0 - shrink_short * t)
        up_bias = 0.0
        dx = w * m_long
        dy = h * m_short
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0 - up_bias * h
        rx1, ry1 = cx - (w/2 + dx), cy - (h/2 + dy)
        rx2, ry2 = cx + (w/2 + dx), cy + (h/2 + dy)

    else:
        # 中性：折中策略
        m_long  = base_long * 0.9 + gain_long * 0.5 * t
        m_short = base_short * (1.0 - 0.4 * t)
        up_bias = up_base * 0.5 * (1.0 if aspect > 1.0 else 0.0)
        # 哪边更长就给哪边用长轴留白
        dx = w * (m_long if w >= h else m_short)
        dy = h * (m_long if h >  w else m_short)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0 - up_bias * h
        rx1, ry1 = cx - (w/2 + dx), cy - (h/2 + dy)
        rx2, ry2 = cx + (w/2 + dx), cy + (h/2 + dy)

    # 最小尺寸保护
    if (rx2 - rx1) < min_side_px:
        c = (rx1 + rx2) / 2.0
        rx1, rx2 = c - min_side_px/2, c + min_side_px/2
    if (ry2 - ry1) < min_side_px:
        c = (ry1 + ry2) / 2.0
        ry1, ry2 = c - min_side_px/2, c + min_side_px/2

    return clamp_rect_to_image(int(rx1), int(ry1), int(rx2), int(ry2), W, H)


# -------------------- 主服务类 --------------------

class HandCamService:
    def __init__(
        self,
        model_path: str,
        root_dir: str = r"D:\5710HAND",
        save_root: Optional[str] = None,
        cam_index: int = 0,
        imgsz: int = 640,
        conf_thr: float = 0.28,
        iou_thr: float = 0.55,
        target_size: Tuple[int, int] = (300, 300),
        pad_color: Tuple[int, int, int] = (255, 255, 255),
        preview_width: int = 1280,
        preview_height: int = 720,
        debug_draw: bool = False,   # 预览叠加可视化
    ):
        self.model_path = model_path
        self.root_dir = root_dir
        self.save_root = save_root or os.path.join(root_dir, r"archive_crops_datasets\live_crops_300")
        self.cam_index = cam_index
        self.imgsz = imgsz
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.target_size = target_size
        self.pad_color = pad_color
        self.preview_w = preview_width
        self.preview_h = preview_height
        self.debug_draw = debug_draw

        ensure_dir(self.save_root)
        self.day_dir = os.path.join(self.save_root, datetime.now().strftime("%Y%m%d"))
        ensure_dir(self.day_dir)

        # 运行态
        self._cap = None
        self._reader_thread = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._latest_raw = None     # 干净源（裁剪/检测用）
        self._latest_frame = None   # 预览帧（可带可视化）
        self._model = None          # 懒加载

        # 最近一次检测（用于预览可视化）
        self._last_boxes: List[List[int]] = []
        self._last_crops: List[List[int]] = []

    # --------------- 生命周期 ---------------

    def start(self):
        """启动摄像头与取帧线程；加载YOLO模型"""
        if self._cap is not None:
            return
        print(f"[INFO] Loading model: {self.model_path}")
        self._model = YOLO(self.model_path)

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[WARN] CAP_DSHOW 打不开，尝试默认后端...")
            cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查 CAM_INDEX 或权限。")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_h)

        self._cap = cap
        self._stop_flag.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        print("[INFO] HandCamService started.")

    def stop(self):
        """停止线程并释放摄像头资源"""
        self._stop_flag.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("[INFO] HandCamService stopped.")

    def _reader_loop(self):
        """后台线程：持续读取最新帧"""
        while not self._stop_flag.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_raw = frame            # 先存干净源
                if self.debug_draw and (self._last_boxes or self._last_crops):
                    vis = frame.copy()
                    # 原始 bbox（绿色）
                    for (x1, y1, x2, y2) in self._last_boxes:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 200, 60), 2)
                    # 自适应矩形裁剪框（蓝色）
                    for (sx1, sy1, sx2, sy2) in self._last_crops:
                        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (220, 120, 0), 2)
                    self._latest_frame = vis
                else:
                    self._latest_frame = frame

    # --------------- 外部接口 ---------------

    def get_preview_frame(self):
        """返回最新预览帧（BGR）；无帧则 None。"""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def detect_once(self, save: bool = True) -> Dict[str, Any]:
        """
        执行一次YOLO检测，按需保存 300x300（等比例+白边）的 ROI 并返回结果。
        """
        with self._lock:
            if self._latest_raw is None:
                return {"ok": False, "has_hand": False, "n": 0, "saved": [], "boxes": [], "crop_boxes": [], "confs": []}
            raw = self._latest_raw.copy()  # 一定用干净源

        H, W = raw.shape[:2]
        res = self._model.predict(
            source=raw,
            imgsz=self.imgsz,
            conf=self.conf_thr,
            iou=self.iou_thr,
            verbose=False
        )[0]

        boxes_obj = res.boxes
        if len(boxes_obj) == 0:
            # 清空预览可视化框
            self._last_boxes = []
            self._last_crops = []
            return {"ok": True, "has_hand": False, "n": 0, "saved": [], "boxes": [], "crop_boxes": [], "confs": [], "timestamp": timestamp()}

        # 解析结果
        xyxy_list, confs = [], []
        for b in boxes_obj:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            xyxy_list.append([x1, y1, x2, y2])
            confs.append(float(b.conf[0]))

        saved_paths: List[str] = []
        crop_boxes: List[List[int]] = []
        ts = timestamp()
        sub_dir = os.path.join(self.day_dir, ts[:8])
        if save:
            ensure_dir(sub_dir)

        # 各向异性自适应裁剪并保存
        for i, (x1, y1, x2, y2) in enumerate(xyxy_list):
            sx1, sy1, sx2, sy2 = rect_expand_anisotropic_adaptive([x1, y1, x2, y2], W, H)
            crop_boxes.append([sx1, sy1, sx2, sy2])

            crop = raw[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            crop_square = resize_with_padding(crop, self.target_size, self.pad_color)

            if save:
                out_path = os.path.join(sub_dir, f"{ts}_{i}.png")
                cv2.imwrite(out_path, crop_square)
                saved_paths.append(out_path)

        # 更新可视化框
        self._last_boxes = xyxy_list if self.debug_draw else []
        self._last_crops = crop_boxes if self.debug_draw else []

        return {
            "ok": True,
            "has_hand": True,
            "n": len(xyxy_list),
            "saved": saved_paths,
            "boxes": xyxy_list,        # 原始 bbox
            "crop_boxes": crop_boxes,  # 最终各向异性裁剪框
            "confs": confs,
            "timestamp": ts
        }


# --------------------- 简易验证主循环（示例） ---------------------
if __name__ == "__main__":
    MODEL = r"D:\5710HAND\runs\hand_det_y11s_768\weights\best.pt"  # 用你最新的权重
    service = HandCamService(
        model_path=MODEL,
        root_dir=r"D:\5710HAND",
        cam_index=0,
        imgsz=640,
        conf_thr=0.28,
        iou_thr=0.55,
        target_size=(300, 300),
        pad_color=(255, 255, 255),
        debug_draw=False   # 需要纯净保存不受影响，因为 detect_once 用的是 _latest_raw
    )

    service.start()
    print("[INFO] Press 'SPACE' to trigger detect_once(); 'ESC' to quit.")

    last_call = 0.0
    interval = 3.0  # 每隔3秒自动检测一次
    try:
        while True:
            frame = service.get_preview_frame()
            if frame is not None:
                cv2.imshow("Preview (anisotropic adaptive crop overlay)", frame)

            # 定时触发一次检测
            now = time.time()
            if now - last_call >= interval:
                result = service.detect_once(save=True)
                print("[CALL detect_once]:", result)
                last_call = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC/Q 退出
                break
            elif key == 32:  # 空格手动触发一次
                result = service.detect_once(save=True)
                print("[SPACE detect_once]:", result)

    finally:
        service.stop()
        cv2.destroyAllWindows()
