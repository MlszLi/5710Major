# batch_grabcut_edges_fast.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ！！！确保同目录下能导入到你这份 hand_grabcut_simple.py
from hand_grabcut_simple import grabcut_hand_edges

# ====== 路径配置 ======
INPUT_ROOT  = Path(r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\Type_01")
OUTPUT_ROOT = Path(r"D:\Files\2025_Y4_S2\AMME5710\Major\Root\grabcut_img")

# 允许的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ====== GrabCut 快速参数（可按需调整）======
# 说明：iter_rect/iter_mask 越小越快；top/left/right/bottom 的比例不要过大
#      否则容易让 mask 里缺少“确定前景/背景”像素引发断言错误（你上次的报错）。
GC_KW = dict(
    top_bg_frac=0.08,
    left_bg_frac=0.08,
    right_bg_frac=0.08,
    bottom_bg_frac=0.06,
    corner_bg_frac=0.15,      # 四角硬编码背景（可 0.10~0.18）
    rect_margin_frac=0.06,
    fg_ellipse_scale=0.80,
    bg_border_frac=0.02,
    iter_rect=1,              # ← 快速：1~2
    iter_mask=1,              # ← 快速：1~2
    grow_frac=0.03,
    show=False
)

# 是否使用 JPG 输出（更快/更小）；False 则沿用原扩展名
USE_JPG_OUTPUT = False
JPG_QUALITY = 90  # 80~92 较合理

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def output_path_for(in_path: Path, out_dir: Path) -> Path:
    if USE_JPG_OUTPUT:
        return out_dir / (in_path.stem + ".jpg")
    else:
        return out_dir / in_path.name

def _worker(in_path_str: str, out_path_str: str, gc_kw: dict, use_jpg: bool, jpg_quality: int):
    """
    子进程执行：对单张图片做 grabcut，并保存到 out_path。
    """
    # 避免 OpenCV 在多进程下开启多线程导致过度竞争
    try:
        import cv2
        cv2.setUseOptimized(True)
        cv2.ocl.setUseOpenCL(False)   # 多进程下一般不用 OpenCL
        cv2.setNumThreads(0)
    except Exception:
        pass

    in_path  = Path(in_path_str)
    out_path = Path(out_path_str)

    # 若已存在则跳过（幂等）
    if out_path.exists():
        return f"SKIP {in_path.name}"

    # 正常调用你已有的函数
    tmp_out = out_path
    if use_jpg and out_path.suffix.lower() != ".jpg":
        # 先用原函数输出到临时 PNG，再转存为 JPG
        tmp_out = out_path.with_suffix(".png")

    grabcut_hand_edges(str(in_path), str(tmp_out), **gc_kw)

    if use_jpg and tmp_out.suffix.lower() == ".png":
        # PNG -> JPG 转存（更快/更小）
        import cv2
        import numpy as np
        img = cv2.imread(str(tmp_out), cv2.IMREAD_UNCHANGED)
        # 如果是带透明通道的 PNG，先铺黑底（或白底）
        if img is not None and img.shape[-1] == 4:
            bgr = img[..., :3]
            alpha = img[..., 3:4] / 255.0
            bg = np.zeros_like(bgr)  # 黑底; 如需白底：np.full_like(bgr, 255)
            img = (bgr * alpha + bg * (1 - alpha)).astype("uint8")
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        try:
            os.remove(str(tmp_out))
        except Exception:
            pass

    return f"DONE {in_path.name}"

def main(max_workers: int = None, chunk_size: int = 32):
    ensure_dir(OUTPUT_ROOT)

    # 收集所有任务
    tasks = []
    for letter in [chr(c) for c in range(ord('A'), ord('Z')+1)]:
        in_dir  = INPUT_ROOT / letter
        out_dir = OUTPUT_ROOT / letter
        if not in_dir.exists():
            print(f"[Skip] {in_dir} 不存在。")
            continue
        ensure_dir(out_dir)

        files = sorted([p for p in in_dir.iterdir() if p.is_file() and is_image(p)])
        if not files:
            print(f"[Info] {in_dir} 没有图片。")
            continue

        print(f"[{letter}] 共 {len(files)} 张，输出目录：{out_dir}")
        for p in files:
            out_p = output_path_for(p, out_dir)
            tasks.append((str(p), str(out_p)))

    if not tasks:
        print("没有可处理的图片。")
        return

    # 并行执行
    worker = partial(_worker, gc_kw=GC_KW, use_jpg=USE_JPG_OUTPUT, jpg_quality=JPG_QUALITY)
    # max_workers 默认：CPU 核心数-1；Windows 下一定要在 __main__ 保护下调用
    if max_workers is None:
        try:
            import os as _os
            cpu = max(1, (_os.cpu_count() or 2) - 1)
        except Exception:
            cpu = 2
        max_workers = cpu

    print(f"并行启动：max_workers={max_workers}，任务数={len(tasks)}")
    done_cnt = 0
    err_cnt = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        # 分批提交，降低一次性提交造成的内存占用
        for i in range(0, len(tasks), chunk_size):
            for in_path_str, out_path_str in tasks[i:i+chunk_size]:
                futures.append(ex.submit(worker, in_path_str, out_path_str))
        for fut in as_completed(futures):
            try:
                msg = fut.result()
                done_cnt += 1
                if done_cnt % 100 == 0:
                    print(f"  进度：{done_cnt}/{len(tasks)}  {msg}")
            except Exception as e:
                err_cnt += 1
                if err_cnt < 20:
                    print(f"  [Error] {e}")
    print(f"✅ 全部完成。成功 {done_cnt-err_cnt}，失败 {err_cnt}。输出：{OUTPUT_ROOT}")

if __name__ == "__main__":
    # Windows 必须加 __main__ 保护；建议先小规模测试（比如只跑 A 目录）
    main()
