# hand_grabcut_simple.py
# -*- coding: utf-8 -*-
"""
GrabCut-based hand segmentation (robust version)

改动要点：
- 四边条改为“可能背景”(GC_PR_BGD)，避免直线硬切手部；
- 在中心加一个“确定前景”(GC_FGD) 的小椭圆种子，稳住GMM建模；
- INIT_WITH_RECT 得到 rough 后，先开再闭再并入 PR_FGD，避免把背景带入；
- 做安全校验（确保FGD/BGD都有像素），防止断言错误；
- 最终保留最大连通域，去掉边角残留；
- 可选下采样加速（downscale_max_side）。

Author: you & ChatGPT
"""
import cv2
import numpy as np
from pathlib import Path

def keep_largest_component(mask255: np.ndarray) -> np.ndarray:
    """保留最大连通域（8连通），去掉小块噪点。"""
    binary = (mask255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return mask255
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    keep = (labels == idx).astype(np.uint8) * 255
    return keep

def grabcut_hand_edges(
    image_path: str,
    save_path: str = "hand_segmented_edges.png",

    # ===== 可调：几何先验 / 迭代 / 速度 =====
    rect_margin_frac: float   = 0.06,   # INIT_WITH_RECT 外框留白比例（相对短边）
    fg_ellipse_scale: float   = 0.80,   # 中心“可能前景”椭圆的尺度（0.7~0.9）
    inner_seed_scale: float   = 0.45,   # 中心“确定前景”小椭圆的尺度（0.35~0.55）

    bg_border_frac: float     = 0.02,   # 最外圈窄边（必为背景，BGD），通常 0.02~0.04
    corner_bg_frac: float     = 0.10,   # 四角方块（必为背景，BGD），0.08~0.18

    top_bg_frac: float        = 0.06,   # 顶部细条（可能背景，PR_BGD），0.04~0.08
    left_bg_frac: float       = 0.06,   # 左侧细条（可能背景，PR_BGD）
    right_bg_frac: float      = 0.06,   # 右侧细条（可能背景，PR_BGD）
    bottom_bg_frac: float     = 0.04,   # 底部细条（可能背景，PR_BGD；手臂常在底部）

    iter_rect: int            = 2,      # INIT_WITH_RECT 迭代次数（1~3）
    iter_mask: int            = 2,      # INIT_WITH_MASK 迭代次数（1~3）

    grow_frac: float          = 0.03,   # 形态学核大小比例（开-闭用），0.015~0.04
    downscale_max_side: int   = 0,      # 可选：下采样加速。<=0 表示不下采样；如 360

    show: bool                = False,  # 是否弹窗显示
    out_with_alpha: bool      = False   # True 则保存 RGBA（前景+透明背景）
):
    # --- 读图 ---
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H0, W0 = img.shape[:2]

    # --- 可选：下采样加速 ---
    scale = 1.0
    if downscale_max_side and max(H0, W0) > downscale_max_side:
        scale = downscale_max_side / float(max(H0, W0))
        img_small = cv2.resize(img, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
    h, w = img_small.shape[:2]

    # --- Step1: 宽松矩形初始化（INIT_WITH_RECT） ---
    m = int(round(rect_margin_frac * min(h, w)))
    rect = (m, m, max(1, w - 2*m), max(1, h - 2*m))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_small, mask, rect, bgdModel, fgdModel, max(1, int(iter_rect)), mode=cv2.GC_INIT_WITH_RECT)

    # 取粗分割
    rough = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # --- Step2: 构建更稳的 trimap ---
    trimap = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # 2.1 最外圈窄边：必为背景（BGD）
    bw, bh = int(bg_border_frac * w), int(bg_border_frac * h)
    if bw > 0:
        trimap[:, :bw]  = cv2.GC_BGD
        trimap[:, -bw:] = cv2.GC_BGD
    if bh > 0:
        trimap[:bh, :]  = cv2.GC_BGD
        trimap[-bh:, :] = cv2.GC_BGD

    # 2.2 中央先验：大椭圆 = 可能前景（PR_FGD）
    cx, cy = w // 2, h // 2
    rx, ry = int(fg_ellipse_scale * w / 2), int(fg_ellipse_scale * h / 2)
    ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(ell, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    trimap[ell == 255] = cv2.GC_PR_FGD

    # 2.2b 中央内圈：小椭圆 = 确定前景（FGD）
    rx2, ry2 = max(5, int(rx * inner_seed_scale)), max(5, int(ry * inner_seed_scale))
    ell2 = np.zeros((h, w), np.uint8)
    cv2.ellipse(ell2, (cx, cy), (rx2, ry2), 0, 0, 360, 255, -1)
    trimap[ell2 == 255] = cv2.GC_FGD

    # 2.3 用粗分割“开-闭”后并入 PR_FGD（避免把背景带入）
    k = 2*max(1, int(round(grow_frac * min(h, w)))) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    rough_refined = cv2.morphologyEx(rough, cv2.MORPH_OPEN,  kernel, iterations=1)
    rough_refined = cv2.morphologyEx(rough_refined, cv2.MORPH_CLOSE, kernel, iterations=1)
    trimap[rough_refined == 255] = cv2.GC_PR_FGD

    # 2.4 四角：必为背景（BGD）
    corner = int(round(corner_bg_frac * min(h, w)))
    if corner > 0:
        trimap[0:corner, 0:corner] = cv2.GC_BGD
        trimap[0:corner, w-corner:w] = cv2.GC_BGD
        trimap[h-corner:h, 0:corner] = cv2.GC_BGD
        trimap[h-corner:h, w-corner:w] = cv2.GC_BGD

    # 2.5 上/左/右/下：可能背景（PR_BGD）——避免直线硬切
    tb = int(round(top_bg_frac * h))
    lb = int(round(left_bg_frac * w))
    rb = int(round(right_bg_frac * w))
    bb = int(round(bottom_bg_frac * h))
    if tb > 0: trimap[:tb, :]     = cv2.GC_PR_BGD
    if lb > 0: trimap[:, :lb]     = cv2.GC_PR_BGD
    if rb > 0: trimap[:, w-rb:]   = cv2.GC_PR_BGD
    if bb > 0: trimap[h-bb:h, :]  = cv2.GC_PR_BGD

    # 2.6 安全校验：确保既有BGD也有FGD
    MIN_SEED = 50
    if (trimap == cv2.GC_BGD).sum() < MIN_SEED:
        trimap[:3, :], trimap[-3:, :] = cv2.GC_BGD, cv2.GC_BGD
        trimap[:, :3], trimap[:, -3:] = cv2.GC_BGD, cv2.GC_BGD
    if (trimap == cv2.GC_FGD).sum() < MIN_SEED:
        trimap[cy-5:cy+6, cx-5:cx+6] = cv2.GC_FGD

    # --- Step3: 用 mask 细化（INIT_WITH_MASK） ---
    bgdModel2 = np.zeros((1, 65), np.float64)
    fgdModel2 = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_small, trimap, None, bgdModel2, fgdModel2, max(1, int(iter_mask)), mode=cv2.GC_INIT_WITH_MASK)

    # 前景 mask
    final_mask_small = np.where((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 平滑 + 最大连通域
    final_mask_small = cv2.morphologyEx(final_mask_small, cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    final_mask_small = keep_largest_component(final_mask_small)

    # --- 回到原尺寸 ---
    if scale != 1.0:
        final_mask = cv2.resize(final_mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
    else:
        final_mask = final_mask_small

    # 合成输出
    if out_with_alpha:
        # RGBA：前景+透明背景
        bgr = img.copy()
        alpha = final_mask
        result = np.dstack([bgr, alpha])
    else:
        # BGR：背景置黑
        result = np.zeros_like(img)
        result[final_mask == 255] = img[final_mask == 255]

    # 保存
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), result)

    # 可视化
    if show:
        dbg = img.copy()
        if corner > 0:
            cv2.rectangle(dbg,(0,0),(corner,corner),(0,0,255),2)
            cv2.rectangle(dbg,(w-corner,0),(w,corner),(0,0,255),2)
            cv2.rectangle(dbg,(0,h-corner),(corner,h),(0,0,255),2)
            cv2.rectangle(dbg,(w-corner,h-corner),(w,h),(0,0,255),2)
        if tb>0: cv2.rectangle(dbg,(0,0),(w,tb),(0,255,255),2)
        if lb>0: cv2.rectangle(dbg,(0,0),(lb,h),(0,255,255),2)
        if rb>0: cv2.rectangle(dbg,(w-rb,0),(w,h),(0,255,255),2)
        if bb>0: cv2.rectangle(dbg,(0,h-bb),(w,h),(0,255,255),2)

        cv2.imshow("Original + forced BG (debug)", dbg)
        cv2.imshow("Mask (final)", final_mask)
        vis = result if result.ndim == 3 else cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        cv2.imshow("Segmented Hand", vis)
        cv2.waitKey(0); cv2.destroyAllWindows()

    return final_mask  # 如需后续管线可直接返回mask

# -----------------------
if __name__ == "__main__":
    # 示例：单张测试
    grabcut_hand_edges(
        image_path=r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\Type_01\C\C_10.jpg",
        save_path=r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\grabcut_img\hand_segmented.png",
        top_bg_frac=0.08, left_bg_frac=0.18, right_bg_frac=0.18, bottom_bg_frac=0.06,
        corner_bg_frac=0.18,
        rect_margin_frac=0.05, fg_ellipse_scale=0.40, inner_seed_scale=0.25,
        bg_border_frac=0.02,
        iter_rect=2, iter_mask=2,
        grow_frac=0.02,
        downscale_max_side=0,   # 例如 360 可显著提速
        show=True,
        out_with_alpha=False
    )
