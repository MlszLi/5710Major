# hand_grabcut_fast.py
# -*- coding: utf-8 -*-
import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Literal

class GrabCutHandSegmenter:
    """
    速度优化版：
      - 首帧/每隔 full_every 帧：全量 GrabCut（GC_INIT_WITH_MASK，iter=full_iters）
      - 其他帧：热启动快速迭代（GC_EVAL，iter=eval_iters）
      - 先缩小到 downscale_size 再运行，最后上采样回原尺寸合成
    """
    def __init__(self,
                 out_bg_color: Tuple[int,int,int]=(0,0,0),
                 downscale_size: int = 200,
                 full_iters: int = 5,
                 eval_iters: int = 1,
                 full_every: int = 5,
                 fg_ellipse_scale: float = 0.62,
                 bg_border_frac: float = 0.08,
                 post_clean: bool = True):
        self.out_bg_color = out_bg_color
        self.downscale_size = int(downscale_size)
        self.full_iters = int(full_iters)
        self.eval_iters = int(eval_iters)
        self.full_every = int(full_every)
        self.fg_ellipse_scale = float(fg_ellipse_scale)
        self.bg_border_frac = float(bg_border_frac)
        self.post_clean = bool(post_clean)

        # 持久化状态（热启动）
        self._mask_small = None
        self._bgModel = None
        self._fgModel = None
        self._frame_count = 0

        try:
            cv2.setUseOptimized(True)
            cv2.ocl.setUseOpenCL(True)
        except Exception:
            pass

    def _build_trimap(self, h: int, w: int) -> np.ndarray:
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)  # 可能背景
        bw = max(1, int(round(self.bg_border_frac * w)))
        bh = max(1, int(round(self.bg_border_frac * h)))
        mask[:bh, :] = cv2.GC_BGD
        mask[-bh:, :] = cv2.GC_BGD
        mask[:, :bw] = cv2.GC_BGD
        mask[:, -bw:] = cv2.GC_BGD

        cx, cy = w // 2, h // 2
        rx = int(self.fg_ellipse_scale * w * 0.5)
        ry = int(self.fg_ellipse_scale * h * 0.5)
        ell = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ell, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        mask[ell == 255] = cv2.GC_PR_FGD
        return mask

    def _morph_clean(self, mask_255: np.ndarray) -> np.ndarray:
        if not self.post_clean:
            return mask_255
        h, w = mask_255.shape[:2]
        k = max(1, int(round(min(h, w) * 0.01)))
        k = k + 1 if k % 2 == 0 else k
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN,  kernel, iterations=1)
        m = cv2.morphologyEx(m,         cv2.MORPH_CLOSE, kernel, iterations=1)
        return m

    def reset_state(self):
        self._mask_small = None
        self._bgModel = None
        self._fgModel = None
        self._frame_count = 0

    def segment(self, roi_bgr: np.ndarray):
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("Empty ROI image.")

        H, W = roi_bgr.shape[:2]
        # 缩放到较小尺寸
        s = self.downscale_size / float(max(H, W))
        if s < 1.0:
            small = cv2.resize(roi_bgr, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        else:
            small = roi_bgr.copy()

        h, w = small.shape[:2]
        self._frame_count += 1
        do_full = (self._mask_small is None) or (self._frame_count % self.full_every == 1)

        if do_full:
            init_mask = self._build_trimap(h, w)
            self._mask_small = init_mask.copy()
            self._bgModel = np.zeros((1, 65), np.float64)
            self._fgModel = np.zeros((1, 65), np.float64)
            mode = cv2.GC_INIT_WITH_MASK
            iters = self.full_iters
        else:
            mode = cv2.GC_EVAL
            iters = self.eval_iters

        mask_gc, self._bgModel, self._fgModel = cv2.grabCut(
            small, self._mask_small, None, self._bgModel, self._fgModel,
            iterCount=int(iters), mode=mode
        )
        self._mask_small = mask_gc

        fg_small = (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD)
        fg_mask_small = (fg_small.astype(np.uint8)) * 255

        fg_mask = cv2.resize(fg_mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        fg_mask = self._morph_clean(fg_mask)

        feat_bgr = np.empty_like(roi_bgr)
        feat_bgr[:] = self.out_bg_color
        feat_bgr[fg_mask == 255] = roi_bgr[fg_mask == 255]
        return feat_bgr, fg_mask


def grabcut_hand_edges(
    image_path: str,
    save_path: str,
    *,
    output: Literal["feat","mask","both"] = "feat",
    downscale_size: int = 200,
    full_iters: int = 5,
    eval_iters: int = 1,
    full_every: int = 5,
    fg_ellipse_scale: float = 0.62,
    bg_border_frac: float = 0.08,
    post_clean: bool = True,
    out_bg_color: Tuple[int,int,int]=(0,0,0),
    mask_suffix: str = "_mask",
    feat_suffix: str = "_fg"
):
    """
    单张图片接口：读取 image_path，运行 GrabCut，保存到 save_path。
    - output='feat'：save_path 保存前景图（置色背景）
    - output='mask'：save_path 保存 0/255 掩膜
    - output='both'：save_path 作为“基名”，保存两份：*_fg.png 和 *_mask.png
    返回：字典，包含已保存文件的路径
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    seg = GrabCutHandSegmenter(
        out_bg_color=out_bg_color,
        downscale_size=downscale_size,
        full_iters=full_iters,
        eval_iters=eval_iters,
        full_every=full_every,
        fg_ellipse_scale=fg_ellipse_scale,
        bg_border_frac=bg_border_frac,
        post_clean=post_clean
    )
    seg.reset_state()
    feat_bgr, fg_mask = seg.segment(img)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    saved = {}
    if output == "feat":
        ok = cv2.imwrite(str(save_path), feat_bgr)
        if not ok: raise IOError(f"Failed to save: {save_path}")
        saved["feat"] = str(save_path)
    elif output == "mask":
        ok = cv2.imwrite(str(save_path), fg_mask)
        if not ok: raise IOError(f"Failed to save: {save_path}")
        saved["mask"] = str(save_path)
    elif output == "both":
        # 以 save_path 的 stem 作为基名
        if save_path.suffix:
            stem = save_path.with_suffix("").name
            out_dir = save_path.parent
        else:
            stem = save_path.name
            out_dir = save_path.parent

        feat_path = out_dir / f"{stem}{feat_suffix}.png"
        mask_path = out_dir / f"{stem}{mask_suffix}.png"
        ok1 = cv2.imwrite(str(feat_path), feat_bgr)
        ok2 = cv2.imwrite(str(mask_path), fg_mask)
        if not ok1 or not ok2:
            raise IOError(f"Failed to save outputs to: {out_dir}")
        saved["feat"] = str(feat_path)
        saved["mask"] = str(mask_path)
    else:
        raise ValueError("output must be one of: 'feat', 'mask', 'both'")

    return saved


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Fast GrabCut hand segmentation (single image)."
    )
    # 必填：输入/输出
    p.add_argument("--image_path", type=str, required=False,
                   help="输入图片路径（单张）。")
    p.add_argument("--save_path", type=str, required=False,
                   help="输出路径。若 output=both，则 save_path 作为基名所在目录/文件名（不需要扩展名）。")

    # 输出模式
    p.add_argument("--output", type=str, default="feat", choices=["feat","mask","both"],
                   help="保存前景图、掩膜或二者。默认 feat。")

    # 可调参数
    p.add_argument("--downscale_size", type=int, default=200)
    p.add_argument("--full_iters", type=int, default=5)
    p.add_argument("--eval_iters", type=int, default=1)
    p.add_argument("--full_every", type=int, default=5)
    p.add_argument("--fg_ellipse_scale", type=float, default=0.62)
    p.add_argument("--bg_border_frac", type=float, default=0.08)
    p.add_argument("--no_clean", action="store_true", help="关闭形态学清理")
    p.add_argument("--bg_color", type=int, nargs=3, default=[0,0,0], help="B G R 背景色")
    p.add_argument("--mask_suffix", type=str, default="_mask")
    p.add_argument("--feat_suffix", type=str, default="_fg")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 如果你想用“硬编码”的方式（与你发的示例一致），就把下面这段启用：
    # ------------------------------------------------------------------
    # 覆盖命令行，直接在此修改路径与参数即可（保留命令行也可用）
    if args.image_path is None and args.save_path is None:
        # 你的示例：单张测试
        saved = grabcut_hand_edges(
            image_path=r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\Type_01\E\E_31.jpg",
            save_path=r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\grabcut_img\hand_segmented.png",
            output="feat",                 # "feat" | "mask" | "both"
            downscale_size=200,
            full_iters=5,
            eval_iters=1,
            full_every=5,
            fg_ellipse_scale=0.62,
            bg_border_frac=0.08,
            post_clean=True,
            out_bg_color=(0,0,0),
            mask_suffix="_mask",
            feat_suffix="_fg"
        )
        print("[OK] Saved:", saved)
        return 0
    # ------------------------------------------------------------------

    # 正常命令行路径
    if args.image_path is None or args.save_path is None:
        parser.error("--image_path 与 --save_path 需要同时提供，或删除二者用上面的硬编码示例。")

    saved = grabcut_hand_edges(
        image_path=args.image_path,
        save_path=args.save_path,
        output=args.output,
        downscale_size=args.downscale_size,
        full_iters=args.full_iters,
        eval_iters=args.eval_iters,
        full_every=args.full_every,
        fg_ellipse_scale=args.fg_ellipse_scale,
        bg_border_frac=args.bg_border_frac,
        post_clean=(not args.no_clean),
        out_bg_color=tuple(args.bg_color),
        mask_suffix=args.mask_suffix,
        feat_suffix=args.feat_suffix
    )
    print("[OK] Saved:", saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
