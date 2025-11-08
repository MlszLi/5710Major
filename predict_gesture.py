# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import argparse, sys
# from pathlib import Path
# import torch, torch.nn as nn
# from torchvision import models
# import numpy as np
# import cv2

# CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# def build_model(num_classes=26):
#     m = models.resnet18(weights=None)
#     m.fc = nn.Linear(m.fc.in_features, num_classes)
#     return m

# def load_model(model_path: str, device: torch.device, num_classes: int = 26):
#     model = build_model(num_classes)
#     state = torch.load(model_path, map_location=device)
#     model.load_state_dict(state, strict=True)
#     model.to(device).eval()
#     return model

# def preprocess_bgr(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
#     """cv2 BGR → RGB → resize → [0,1] → [1,3,H,W]"""
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
#     img_rgb /= 255.0
#     x = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).contiguous()
#     return x

# @torch.inference_mode()
# def predict_bgr(model, device, img_bgr: np.ndarray, size=224, topk=3):
#     x = preprocess_bgr(img_bgr, size=size).to(device)
#     logits = model(x)
#     probs = torch.softmax(logits, dim=1)[0]
#     confs, ids = torch.topk(probs, k=min(topk, len(CLASSES)))
#     confs = confs.detach().cpu().numpy().tolist()
#     ids   = ids.detach().cpu().numpy().tolist()
#     return [(CLASSES[i], float(c)) for i,c in zip(ids, confs)]

# # ====== 保留原来的命令行用法（兼容旧流程） ======
# def _collect_images(inp: str):
#     p = Path(inp)
#     exts = {".jpg",".jpeg",".png",".bmp",".webp"}
#     if p.is_file(): return [str(p)]
#     if p.is_dir():  return [str(x) for x in sorted(p.rglob("*")) if x.suffix.lower() in exts]
#     raise FileNotFoundError(f"Input not found: {inp}")

# def main():
#     ap = argparse.ArgumentParser(description="Predict A–Z hand gesture with trained ResNet-18.")
#     ap.add_argument("--model", required=True, help="Path to .pt weights")
#     ap.add_argument("--input", required=True, help="Image file or a folder")
#     ap.add_argument("--size", type=int, default=224)
#     ap.add_argument("--topk", type=int, default=3)
#     ap.add_argument("--cpu", action="store_true")
#     args = ap.parse_args()

#     device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
#     print(f"[INFO] Using device: {device}")

#     model = load_model(args.model, device, num_classes=len(CLASSES))
#     paths = _collect_images(args.input)
#     print(f"[INFO] Found {len(paths)} image(s).")

#     for p in paths:
#         img = cv2.imread(p); 
#         if img is None: 
#             print(f"[WARN] fail to read {p}", file=sys.stderr); 
#             continue
#         preds = predict_bgr(model, device, img, size=args.size, topk=args.topk)
#         top = preds[0]
#         top_str = ", ".join([f"{c}:{s:.3f}" for c,s in preds])
#         print(f"{p}\n  → Top-1: {top[0]} ({top[1]:.3f}) | Top-{len(preds)}: {top_str}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models
import numpy as np
import cv2

# === 类别表：A-Z ===
CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------- 模型构建（可选带 Dropout 的头） ----------
def build_model(num_classes=26, with_dropout=False, p=0.5):
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    if with_dropout:
        m.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_f, num_classes)
        )
    else:
        m.fc = nn.Linear(in_f, num_classes)
    return m

# ---------- 兼容加载（自动识别头部形态） ----------
def load_model(model_path: str, device: torch.device, num_classes: int = 26):
    # 安全加载（避免 pickle 任意执行）; 旧版 PyTorch 可能不支持 weights_only，可按需去掉
    state = torch.load(model_path, map_location=device, weights_only=True)

    # 1) 识别最后层键名，判断是否含 Dropout 头
    keys = list(state.keys())
    has_dropout_head = any(k.startswith("fc.1.") for k in keys)  # Sequential(Dropout, Linear)
    linear_head      = any(k.startswith("fc.weight") for k in keys)  # 单层 Linear

    # 2) 构建匹配的结构
    if has_dropout_head:
        model = build_model(num_classes=num_classes, with_dropout=True, p=0.5)
    else:
        model = build_model(num_classes=num_classes, with_dropout=False)

    # 3) 柔性加载：如果你确认键完全匹配，可以 strict=True；为稳妥这里用 False
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # 打印一下方便排查（非致命）
        print("[load_model] missing keys:", missing)
        print("[load_model] unexpected keys:", unexpected)

    model.to(device).eval()
    return model

# ---------- 预处理：与训练对齐（Resize+RGB+[0,1]+ImageNet标准化） ----------
def preprocess_bgr(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # ImageNet 归一化
    img_rgb[..., 0] = (img_rgb[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    img_rgb[..., 1] = (img_rgb[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    img_rgb[..., 2] = (img_rgb[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    x = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return x

@torch.inference_mode()
def predict_bgr(model, device, img_bgr: np.ndarray, size=224, topk=3):
    x = preprocess_bgr(img_bgr, size=size).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    k = min(topk, len(CLASSES))
    confs, ids = torch.topk(probs, k=k)
    confs = confs.detach().cpu().numpy().tolist()
    ids   = ids.detach().cpu().numpy().tolist()
    return [(CLASSES[i], float(c)) for i,c in zip(ids, confs)]

# ====== 命令行用法（保留） ======
def _collect_images(inp: str):
    p = Path(inp)
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    if p.is_file(): return [str(p)]
    if p.is_dir():  return [str(x) for x in sorted(p.rglob("*")) if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Input not found: {inp}")

def main():
    ap = argparse.ArgumentParser(description="Predict A–Z hand gesture with trained ResNet-18.")
    ap.add_argument("--model", required=True, help="Path to .pt weights")
    ap.add_argument("--input", required=True, help="Image file or a folder")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"[INFO] Using device: {device}")

    model = load_model(args.model, device, num_classes=len(CLASSES))
    paths = _collect_images(args.input)
    print(f"[INFO] Found {len(paths)} image(s).")

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] fail to read {p}", file=sys.stderr)
            continue
        preds = predict_bgr(model, device, img, size=args.size, topk=args.topk)
        top = preds[0]
        top_str = ", ".join([f"{c}:{s:.3f}" for c,s in preds])
        print(f"{p}\n  → Top-1: {top[0]} ({top[1]:.3f}) | Top-{len(preds)}: {top_str}")

if __name__ == "__main__":
    main()
