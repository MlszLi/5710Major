# preprocess_hand_svm.py  ——  FAST: Seeded KMeans (default) / fallback SVM
import numpy as np, cv2
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
except Exception as e:
    raise RuntimeError("请先安装 scikit-learn:  pip install scikit-learn") from e


# ============ 内部工具 ============

def _features5(img_bgr: np.ndarray) -> np.ndarray:
    """像素特征: [H,S,V, Cr, Cb] -> float32, shape(H,W,5)"""
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ycc  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Cr, Cb = ycc[...,1], ycc[...,2]
    feat = np.dstack([hsv[...,0], hsv[...,1], hsv[...,2], Cr, Cb]).astype(np.float32)
    return feat  # H,W,5

def _inner_box(x1,y1,x2,y2, ratio=0.18):
    w,h = x2-x1, y2-y1
    dx,dy = int(w*ratio), int(h*ratio)
    return x1+dx, y1+dy, x2-dx, y2-dy

def _clip_box(x1,y1,x2,y2, W,H):
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)

def _largest_cc(mask: np.ndarray) -> np.ndarray:
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1: return mask
    comp = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == comp).astype(np.uint8) * 255


# ============ 可复用状态（缓存上帧的聚类中心/分类器） ============
@dataclass
class PreprocState:
    kmeans_fg: Optional[np.ndarray] = None  # (5,) 上一帧前景中心
    kmeans_bg: Optional[np.ndarray] = None  # (5,) 上一帧背景中心
    svm_clf:   Optional[object]     = None  # 备用：上一帧 SVM

STATE = PreprocState()


# ============ 主入口：更快的“种子化 KMeans” ============

def refine_hand_roi_fast(
    frame_bgr: np.ndarray, xyxy, *,
    down: float = 0.6,        # 手框内下采样比例（0.5~0.7 更快）
    fg_inset: float = 0.14,   # 前景种子内缩
    margin: float = 0.08,     # 最终外扩 margin，避免“瘦手”
    kmeans_iter: int = 20,    # KMeans 迭代次数
    reuse_centers: bool = True,
    safety_open: int = 3,     # 开运算核
    safety_close: int = 5,    # 闭运算核
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
    """
    返回: (roi_bgr, roi_mask, refined_bbox)；失败返回 (None, None, None)
    """
    H,W = frame_bgr.shape[:2]
    x1,y1,x2,y2 = map(int, xyxy)
    x1,y1,x2,y2 = _clip_box(x1,y1,x2,y2, W,H)
    if x2<=x1 or y2<=y1: return None,None,None

    patch = frame_bgr[y1:y2, x1:x2].copy()
    ph,pw = patch.shape[:2]
    if down < 1.0:
        small = cv2.resize(patch, (max(16,int(pw*down)), max(16,int(ph*down))), interpolation=cv2.INTER_AREA)
    else:
        small = patch

    sh,sw = small.shape[:2]
    feat  = _features5(small)      # H,W,5
    Xall  = feat.reshape(-1,5)

    # 前景种子：内缩框；背景种子：四周环带
    ix1,iy1,ix2,iy2 = _inner_box(0,0,sw,sh, ratio=fg_inset)
    fg_seed = np.zeros((sh,sw), np.uint8); fg_seed[iy1:iy2, ix1:ix2] = 1
    edge = max(2, int(min(sw,sh)*0.12))
    bg_seed = np.zeros((sh,sw), np.uint8)
    bg_seed[:edge,:]=1; bg_seed[-edge:,:]=1; bg_seed[:,:edge]=1; bg_seed[:,-edge:]=1

    # 计算种子的均值作为初始中心
    fg_idx = np.where(fg_seed.reshape(-1)==1)[0]
    bg_idx = np.where(bg_seed.reshape(-1)==1)[0]
    if len(fg_idx)==0 or len(bg_idx)==0: return None,None,None
    fg_center = Xall[fg_idx].mean(axis=0)
    bg_center = Xall[bg_idx].mean(axis=0)

    # 复用上一帧中心，提升稳定+速度
    init_centers = np.stack([bg_center, fg_center], axis=0)
    if reuse_centers and (STATE.kmeans_fg is not None) and (STATE.kmeans_bg is not None):
        init_centers = np.stack([STATE.kmeans_bg, STATE.kmeans_fg], axis=0)

    # KMeans（k=2）
    km = KMeans(n_clusters=2, n_init=1, max_iter=kmeans_iter, init=init_centers, random_state=0)
    labels = km.fit_predict(Xall).reshape(sh,sw)
    # 按“靠近 fg_center 的簇为前景”
    fg_id = np.argmin([np.linalg.norm(km.cluster_centers_[i]-fg_center) for i in (0,1)])
    mask = (labels==fg_id).astype(np.uint8)*255

    # 保存中心，供下帧复用
    STATE.kmeans_bg = km.cluster_centers_[1-fg_id]
    STATE.kmeans_fg = km.cluster_centers_[fg_id]

    # 形态学 & 最大连通域
    if safety_open>0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((safety_open,safety_open), np.uint8))
    if safety_close>0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((safety_close,safety_close), np.uint8))
    mask = _largest_cc(mask)

    # 还原到原 patch 尺寸
    if down < 1.0:
        mask = cv2.resize(mask, (pw,ph), interpolation=cv2.INTER_NEAREST)

    ys,xs = np.where(mask>0)
    if len(xs)==0: return None,None,None
    x_min,x_max = xs.min(), xs.max(); y_min,y_max = ys.min(), ys.max()

    # 安全外扩，避免“瘦手”
    bw, bh = x_max-x_min+1, y_max-y_min+1
    mx, my = int(bw*margin), int(bh*margin)
    x_min = max(0, x_min-mx); y_min = max(0, y_min-my)
    x_max = min(pw-1, x_max+mx); y_max = min(ph-1, y_max+my)

    roi = patch[y_min:y_max+1, x_min:x_max+1]
    refined_bbox = (x1+x_min, y1+y_min, x1+x_max+1, y1+y_max+1)
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
    return roi, roi_mask, refined_bbox


# ============ 备选：SVM（保持接口） ============

def refine_hand_roi_svm(
    frame_bgr: np.ndarray, xyxy, *,
    down: float = 0.6, fg_inset: float = 0.18, C: float = 1.0, kernel='linear'
):
    """保留旧接口，必要时可切回 SVM。"""
    H,W = frame_bgr.shape[:2]
    x1,y1,x2,y2 = map(int, xyxy)
    x1,y1,x2,y2 = _clip_box(x1,y1,x2,y2, W,H)
    if x2<=x1 or y2<=y1: return None,None,None
    patch = frame_bgr[y1:y2, x1:x2].copy()
    ph,pw = patch.shape[:2]
    small = cv2.resize(patch, (max(16,int(pw*down)), max(16,int(ph*down))), interpolation=cv2.INTER_AREA) if down<1.0 else patch
    sh,sw = small.shape[:2]
    feat   = _features5(small)
    # seeds
    ix1,iy1,ix2,iy2 = _inner_box(0,0,sw,sh, ratio=fg_inset)
    fg = np.zeros((sh,sw),np.uint8); fg[iy1:iy2, ix1:ix2]=1
    edge = max(2, int(min(sw,sh)*0.12))
    bg = np.zeros((sh,sw),np.uint8); bg[:edge,:]=1; bg[-edge:,:]=1; bg[:,:edge]=1; bg[:,-edge:]=1
    ys_f,xs_f = np.where(fg==1); ys_b,xs_b = np.where(bg==1)
    def sample(xs, ys, n=400):
        if len(xs)>n:
            idx = np.random.choice(len(xs), n, replace=False); return xs[idx], ys[idx]
        return xs, ys
    xs_f,ys_f = sample(xs_f,ys_f); xs_b,ys_b = sample(xs_b,ys_b)
    Xf = feat[ys_f, xs_f, :]; Xb = feat[ys_b, xs_b, :]
    X  = np.vstack([Xf, Xb]); y = np.hstack([np.ones(len(Xf)), np.zeros(len(Xb))])
    clf = SVC(C=C, kernel=kernel, gamma='scale')
    clf.fit(X,y)
    yhat = clf.predict(feat.reshape(-1, feat.shape[2])).reshape(sh,sw).astype(np.uint8)
    if down<1.0: yhat = cv2.resize(yhat,(pw,ph),interpolation=cv2.INTER_NEAREST)
    mask = (yhat*255).astype(np.uint8)
    mask = cv2.medianBlur(mask,5)
    mask = _largest_cc(mask)
    ys,xs = np.where(mask>0)
    if len(xs)==0: return None,None,None
    x_min,x_max = xs.min(), xs.max(); y_min,y_max = ys.min(), ys.max()
    roi = patch[y_min:y_max+1, x_min:x_max+1]
    return roi, mask[y_min:y_max+1, x_min:x_max+1], (x1+x_min, y1+y_min, x1+x_max+1, y1+y_max+1)
