import argparse
import csv
import os
import pickle
import ast
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import clip  # installed via the repo's environment.yml


# -------------------------------
# Image I/O and normalization
# -------------------------------
def load_padchest_image(path: str) -> Image.Image:
    """
    Loads a PadChest image, applies percentile-based normalization,
    and returns it as an RGB PIL Image.
    """
    img = Image.open(path).convert("L")  # Ensure grayscale
    arr = np.array(img, dtype=np.float32)

    # Normalize by 99th percentile to handle bright outliers
    p99 = np.percentile(arr, 99)
    arr = np.clip(arr / (p99 + 1e-8), 0, 1)

    # Convert back to 8-bit image
    arr8 = (arr * 255).astype(np.uint8)

    # Stack into 3 channels (RGB)
    rgb = np.stack([arr8] * 3, axis=-1)

    return Image.fromarray(rgb)


# -------------------------------
# CSV reading + boxes parsing
# -------------------------------
def _parse_boxes_field(s: str) -> List[List[float]]:
    """
    Parses the 'boxes' CSV column.

    Expected structure (stringified):
        [
          [label_id (int), caption (str), [[x1,y1,x2,y2], [..], ...]],
          [ ... ],
          ...
        ]

    Coordinates are assumed normalized in [0,1] relative to image width/height.
    Returns a flat list of boxes [[x1,y1,x2,y2], ...]. If none, returns [].
    """
    if not s:
        return []

    # Try safe Python literal parsing first (handles CSV-escaped quotes well)
    try:
        data = ast.literal_eval(s)
    except Exception:
        # Last resort: strip whitespace, fallback empty
        return []

    boxes: List[List[float]] = []
    # Some files may give just one triplet (not wrapped in list); normalize to list
    if isinstance(data, (list, tuple)) and len(data) == 3 and isinstance(data[2], (list, tuple)):
        data = [data]

    if not isinstance(data, (list, tuple)):
        return []

    for item in data:
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        candidate = item[2]
        if isinstance(candidate, (list, tuple)):
            for bb in candidate:
                if (
                    isinstance(bb, (list, tuple))
                    and len(bb) == 4
                    and all(isinstance(v, (int, float)) for v in bb)
                ):
                    boxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
    return boxes


def read_csv_rows(csv_path: str) -> List[Tuple[str, str, List[List[float]]]]:
    """
    Returns list of (image_name, caption_string, boxes_list).
    Expects columns: ImageID, report_en, boxes
    """
    rows: List[Tuple[str, str, List[List[float]]]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = (r.get("ImageID") or "").strip()
            cap = (r.get("report_en") or "").strip()
            boxes_str = r.get("boxes") or ""
            boxes = _parse_boxes_field(boxes_str.strip())
            if img:
                rows.append((img, cap, boxes))
    return rows


# -------------------------------
# Mask building
# -------------------------------
def build_mask_from_boxes(image_size: Tuple[int, int], boxes: List[List[float]]) -> Image.Image:
    """
    Given (W, H) and a list of normalized boxes [x1,y1,x2,y2] in [0,1],
    returns a binary mask PIL Image (mode 'L'), with 255 inside boxes and 0 outside.
    """
    W, H = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    if not boxes:
        return Image.fromarray(mask)

    for (x1, y1, x2, y2) in boxes:
        # Convert normalized to pixel coordinates; clamp to image bounds
        px1 = max(0, min(W, int(round(x1 * W))))
        py1 = max(0, min(H, int(round(y1 * H))))
        px2 = max(0, min(W, int(round(x2 * W))))
        py2 = max(0, min(H, int(round(y2 * H))))
        if px2 <= px1 or py2 <= py1:
            continue
        mask[py1:py2, px1:px2] = 255

    return Image.fromarray(mask)


# -------------------------------
# CLIP encoding
# -------------------------------
@torch.no_grad()
def encode_images_with_clip(
    rows: List[Tuple[str, str, List[List[float]]]],
    images_root: str,
    clip_model_type: str = "ViT-B/32",
    device: Optional[str] = None,
    normalize: bool = True,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Encodes images and their mask-from-boxes with CLIP. Returns:
      embs: list[np.float32]  (contains BOTH image and mask embeddings)
      caps: list[{
          'image_id': str,
          'caption': str,
          'boxes': List[List[float]],          # normalized boxes as read
          'clip_embedding': int,               # index of image embedding in 'embeddings'
          'clip_mask_embedding': Optional[int] # index of mask embedding in 'embeddings' (always present)
      }]

    The embeddings list appends per-sample:
      [image_emb, mask_emb, image_emb, mask_emb, ...]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(clip_model_type, device=device)
    model.eval()

    embeddings: List[np.ndarray] = []
    captions: List[Dict[str, Any]] = []

    idx = 0
    for (img_name, text, boxes) in tqdm(rows, ncols=100):
        img_path = os.path.join(images_root, img_name)
        if not os.path.exists(img_path):
            continue

        # Load normalized RGB image
        try:
            image_rgb = load_padchest_image(img_path)
        except Exception:
            continue

        # --- Build mask from boxes (same WxH) then make it RGB for CLIP preprocess ---
        W, H = image_rgb.size
        mask_L = build_mask_from_boxes((W, H), boxes)
        mask_rgb = Image.merge("RGB", (mask_L, mask_L, mask_L))

        # --- Preprocess both ---
        image_input = preprocess(image_rgb).unsqueeze(0).to(device)
        mask_input = preprocess(mask_rgb).unsqueeze(0).to(device)

        # --- Encode both ---
        image_features = model.encode_image(image_input)  # (1, D)
        mask_features = model.encode_image(mask_input)    # (1, D)

        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            mask_features  = mask_features  / mask_features.norm(dim=-1, keepdim=True)

        img_vec = image_features[0].detach().cpu().numpy().astype(np.float32)
        msk_vec = mask_features[0].detach().cpu().numpy().astype(np.float32)

        # Append to the shared embeddings pool
        embeddings.append(img_vec)
        clip_idx = idx
        idx += 1

        embeddings.append(msk_vec)
        clip_mask_idx = idx
        idx += 1

        captions.append({
            "image_id": img_name,
            "caption": text or "",
            "boxes": boxes,  # keep normalized coords
            "clip_embedding": clip_idx,
            "clip_mask_embedding": clip_mask_idx
        })

    return embeddings, captions


# -------------------------------
# Saving
# -------------------------------
def save_pkl(embs: List[np.ndarray], caps: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    obj = {
        "clip_embedding": embs,   # includes both image + mask embeddings
        "captions": caps,         # each item has indices for both
    }
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {len(caps)} samples ({len(embs)} embeddings) -> {out_path}")


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=str, default="data/padchestgr/PadChest_GR",
                    help="Folder with all PadChest_GR images")
    ap.add_argument("--train_csv", type=str, default="data/padchestgr/train_final_separated.csv")
    ap.add_argument("--test_csv",  type=str, default="data/padchestgr/test_final_separated.csv")
    ap.add_argument("--out_train", type=str, default="data/padchestgr/padchestgr_clip_ViT-B_32_train_PADCHESTGR_REGION_INDEX.pkl")
    ap.add_argument("--out_test",  type=str, default="data/padchestgr/padchestgr_clip_ViT-B_32_test_PADCHESTGR_REGION_INDEX.pkl")
    ap.add_argument("--clip_model_type", type=str, default="ViT-B/32",
                    choices=["ViT-B/32", "ViT-B/16", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"])
    ap.add_argument("--no_normalize", action="store_true", help="Disable L2-normalization of CLIP features")
    args = ap.parse_args()

    normalize = not args.no_normalize

    # ---------- TRAIN ----------
    train_rows = read_csv_rows(args.train_csv)
    # Optionally drop empty captions for training:
    train_rows = [(i, c, b) for (i, c, b) in train_rows if c and c.strip()]

    train_embs, train_caps = encode_images_with_clip(
        train_rows, args.images_root, clip_model_type=args.clip_model_type, normalize=normalize
    )
    save_pkl(train_embs, train_caps, args.out_train)

    # ---------- TEST ----------
    test_rows = read_csv_rows(args.test_csv)  # may contain empty captions; keep as-is
    test_embs, test_caps = encode_images_with_clip(
        test_rows, args.images_root, clip_model_type=args.clip_model_type, normalize=normalize
    )
    save_pkl(test_embs, test_caps, args.out_test)


if __name__ == "__main__":
    main()
