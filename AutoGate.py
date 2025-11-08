import numpy as np
from collections import deque

# --------- MediaPipe indices and topology ----------
WRIST=0
MCP_I, MCP_M, MCP_R, MCP_P = 5, 9, 13, 17
TIP_IDS = [4, 8, 12, 16, 20]  # thumb→pinky tips
ANCHORS = [WRIST, MCP_I, MCP_M, MCP_R, MCP_P]
EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# -------------- small helpers --------------
def _nz(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def normalize_scale(L):
    """Translate wrist to origin and divide by |wrist->middle MCP|. Returns P (21,3), hand_size(px)."""
    P = L.astype(np.float32).copy()
    P -= P[WRIST]
    s = float(np.linalg.norm(P[MCP_M])) or 1.0
    return P / s, s

def kabsch(A, B):
    """Rigid alignment A->B (no scale)."""
    cA, cB = A.mean(0), B.mean(0)
    A0, B0 = A - cA, B - cB
    U,S,Vt = np.linalg.svd(A0.T @ B0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R.astype(np.float32), t.astype(np.float32)

def validate_hand(landmarks_px, img_w, img_h, min_box_frac=0.03):
    xs = np.array([p[0] for p in landmarks_px])
    ys = np.array([p[1] for p in landmarks_px])

    # (a) bbox should be large enough
    bw, bh = xs.max()-xs.min(), ys.max()-ys.min()
    if bw*bh < min_box_frac * (img_w*img_h):
        return False

    # (b) distances wrist→tips should be reasonable
    wrist = np.array(landmarks_px[0])
    d_tips = np.linalg.norm(np.array([landmarks_px[i] for i in TIP_IDS]) - wrist, axis=1)
    if np.mean(d_tips) < 0.07 * max(img_w, img_h):   # too tiny → likely wrong projection
        return False

    # (c) per-finger monotonicity (wrist→MCP→PIP→DIP→TIP distances increase)
    FINGERS = {
        "index": [0, 5, 6, 7, 8],
        "middle":[0, 9,10,11,12],
        "ring":  [0,13,14,15,16],
        "pinky": [0,17,18,19,20],
        "thumb": [0,1,2,3,4],
    }
    for seq in FINGERS.values():
        d = [np.linalg.norm(np.array(landmarks_px[i]) - np.array(landmarks_px[0])) for i in seq]
        if not all(d[i] <= d[i+1] + 4 for i in range(len(d)-1)):  # +4px tolerance
            return False
    return True


def palm_area_norm(Pn):
    """Area of MCP quadrilateral (scale-invariant since Pn is normalized)."""
    idx, mid, ring, pink = Pn[MCP_I,:2], Pn[MCP_M,:2], Pn[MCP_R,:2], Pn[MCP_P,:2]
    quad = np.stack([idx, mid, ring, pink])
    return 0.5 * abs(np.dot(quad[:,0], np.roll(quad[:,1], -1)) -
                     np.dot(quad[:,1], np.roll(quad[:,0], -1)))

def ransac_palm_residual(Pn, Tn, iters=60, thresh=0.06, rng=np.random):
    """RANSAC on anchors (gesture-agnostic). Returns median residual and inlier count."""
    idx = np.array(ANCHORS)
    best_inliers = None
    best_R = np.eye(3, dtype=np.float32); best_t = np.zeros(3, dtype=np.float32)

    for _ in range(iters):
        samp = rng.choice(idx, size=3, replace=False)
        R,t = kabsch(Pn[samp], Tn[samp])
        P_try = (Pn @ R.T) + t
        res = np.linalg.norm(P_try[idx] - Tn[idx], axis=1)
        inl = res < thresh
        if (best_inliers is None) or (inl.sum() > best_inliers.sum()):
            best_inliers, best_R, best_t = inl, R, t

    if best_inliers is None or best_inliers.sum() < 4:
        return np.inf, 0  # bad alignment
    # refine on inliers
    R,t = kabsch(Pn[idx][best_inliers], Tn[idx][best_inliers])
    P_fit = (Pn @ R.T) + t
    res = np.linalg.norm(P_fit[idx] - Tn[idx], axis=1)
    return float(np.median(res)), int(best_inliers.sum())

# -------- Running stats with inverse-IQR weighting --------
class RunningIQR:
    def __init__(self, maxlen=512):
        self.buf = deque(maxlen=maxlen)
    def update(self, x):
        self.buf.append(float(x))
    def median_iqr(self):
        if len(self.buf) < 8:
            return None, None, None  # not enough data
        arr = np.array(self.buf, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25,75])
        iqr = float(q3 - q1) if q3 > q1 else 1e-6
        med = float(np.median(arr))
        return med, float(q1), float(q3)
    def weight(self):
        # inverse IQR (stability) normalized later
        med, q1, q3 = self.median_iqr()
        if med is None: return 0.0
        iqr = q3 - q1
        return 1.0 / (iqr + 1e-6)

class RANSACRejectionGate:
    """
    Self-adaptive RANSAC gate with detailed debug output.
    Set `debug=True` when you initialize it to print intermediate values.
    """
    def __init__(self, win=512, ransac_iters=60, ransac_thresh=0.06,
                 k_score=3.0, close_ratio=0.05, min_inliers=4,
                 template_palm=None, debug=False,warmup_n=32):
        self.warmup_n = int(warmup_n)            # ← store warm-up length
        self.ransac_iters = ransac_iters
        self.ransac_thresh = ransac_thresh
        self.k_score = float(k_score)
        self.close_ratio = float(close_ratio)
        self.min_inliers = int(min_inliers)
        self.template = template_palm.astype(np.float32) if template_palm is not None else None
        self.debug = debug  # enable/disable debug output

        # running stats for score components
        self.stat_palm = RunningIQR(win)
        self.stat_area = RunningIQR(win)
        self.stat_score = RunningIQR(win)
        self.stat_bones = [RunningIQR(win) for _ in EDGES]

    def _init_template_if_needed(self, Pn):
        if self.template is None:
            self.template = Pn.copy()
            print("[INIT] Template palm initialized")

    def _too_close(self, L, hand_size_px):
        min_d = self.close_ratio * max(hand_size_px, 1.0)
        for (i, j) in EDGES:
            d = np.linalg.norm(L[j] - L[i])
            if d < min_d:
                # print(f"[REJECT] Too close: edge ({i},{j}) distance={d:.3f} < {min_d:.3f}")
                return True
        return False

    def is_valid(self, L21x3):
        if not isinstance(L21x3, np.ndarray) or L21x3.shape != (21,3) or not np.isfinite(L21x3).all():
            # print("[REJECT] Invalid or non-finite input shape")
            return False
        
        # Normalize
        Pn, hand_size_px = normalize_scale(L21x3)
        if hand_size_px < 1e-6:
            # print("[REJECT] Hand size too small or zero")
            return False

        # Hard reject: collapsed adjacent points
        if self._too_close(L21x3, hand_size_px):
            return False

        # Template + RANSAC residual
        self._init_template_if_needed(Pn)
        Tn = self.template
        palm_res, inliers = ransac_palm_residual(Pn, Tn, iters=self.ransac_iters, thresh=self.ransac_thresh)
        # print(f"[DEBUG] palm_res={palm_res:.5f}, inliers={inliers}")
        if not np.isfinite(palm_res) or inliers < self.min_inliers:
            # print(f"[REJECT] RANSAC failed: inliers={inliers} < {self.min_inliers}")
            return False

        # MCP area
        area = palm_area_norm(Pn)
        # print(f"[DEBUG] MCP area={area:.5f}")

        # Bone ratios
        bone_ratios = np.array([np.linalg.norm(Pn[j]-Pn[i]) for (i,j) in EDGES], dtype=np.float32)
        # print(f"[DEBUG] Mean bone ratio={bone_ratios.mean():.5f}")

        # Weights (inverse IQR)
        w_palm = self.stat_palm.weight()
        w_area = self.stat_area.weight()
        w_bone = np.array([s.weight() for s in self.stat_bones], dtype=np.float32)
        if (w_palm + w_area + np.sum(w_bone)) == 0.0:
            w_palm, w_area = 1.0, 1.0
            w_bone = np.ones(len(EDGES), dtype=np.float32) * 0.5
        w_sum = w_palm + w_area + float(np.sum(w_bone))
        w_palm /= w_sum; w_area /= w_sum; w_bone /= w_sum

        # Bone deviation score
        bone_dev = 0.0
        for k, r in enumerate(bone_ratios):
            med, q1, q3 = self.stat_bones[k].median_iqr()
            if med is None:
                continue
            iqr = (q3 - q1) or 1e-6
            if r < q1:
                bone_dev += (q1 - r) / iqr * w_bone[k]
            elif r > q3:
                bone_dev += (r - q3) / iqr * w_bone[k]
        # print(f"[DEBUG] bone_dev={bone_dev:.5f}")

        # Palm & area deviation
        med_p, q1_p, q3_p = self.stat_palm.median_iqr()
        med_a, q1_a, q3_a = self.stat_area.median_iqr()

        palm_dev = 0.0
        if med_p is not None:
            iqr_p = (q3_p - q1_p) or 1e-6
            if palm_res < q1_p:
                palm_dev = (q1_p - palm_res) / iqr_p * w_palm
            elif palm_res > q3_p:
                palm_dev = (palm_res - q3_p) / iqr_p * w_palm

        area_dev = 0.0
        if med_a is not None:
            iqr_a = (q3_a - q1_a) or 1e-6
            if area < q1_a:
                area_dev = (q1_a - area) / iqr_a * w_area
            elif area > q3_a:
                area_dev = (area - q3_a) / iqr_a * w_area

        # ---- Composite score (larger => more outlier-ish)
        score = palm_dev + area_dev + bone_dev
        # print(f"[DEBUG] score={score:.5f}, palm_dev={palm_dev:.5f}, area_dev={area_dev:.5f}")

        # ---- Adaptive threshold with warm-up
        history = len(self.stat_score.buf)
        if history < self.warmup_n:
            ok = True
            # print(f"[WARMUP] Accepting frame ({history}/{self.warmup_n}) to build stats")
        else:
            med_s, q1_s, q3_s = self.stat_score.median_iqr()
            # guard against degenerate IQR; widen a little during early post-warmup
            iqr_s = max((q3_s - q1_s), 1e-6)
            # optional cushion for first few post-warmup frames
            cushion = 1.25 if history < (self.warmup_n + 16) else 1.0
            bound = med_s + cushion * self.k_score * iqr_s
            ok = (score <= bound)
            # print(f"[DEBUG] threshold={bound:.5f}, median={med_s:.5f}, IQR={iqr_s:.5f}, hist={history}")

        # ---- Update stats on accept
        if ok:
            self.stat_palm.update(palm_res)
            self.stat_area.update(area)
            self.stat_score.update(score)
            for k, r in enumerate(bone_ratios):
                self.stat_bones[k].update(float(r))

        return ok
