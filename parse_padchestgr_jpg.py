import argparse
import csv
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

def load_padchest_image(path: str) -> Image.Image:
    """
    Loads a PadChest image, applies percentile-based normalization,
    and returns it as an RGB PIL Image.
    """
    # Open image (likely grayscale DICOM converted to PNG/JPG)
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


import clip  # installed via the repo's environment.yml


def read_csv_pairs(csv_path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (image_name, caption_string). Missing captions become "".
    Expects columns: ImageID, report_en
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = (r.get("ImageID") or "").strip()
            cap = (r.get("report_en") or "").strip()
            if img:
                rows.append((img, cap))
    return rows


@torch.no_grad()
def encode_images_with_clip(
    pairs: List[Tuple[str, str]],
    images_root: str,
    clip_model_type: str = "ViT-B/32",
    device: Optional[str] = None,
    normalize: bool = True,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Encodes images and returns:
      embs: list[np.float32 vector]
      caps: list[{'image_id': id_or_name, 'caption': str, 'clip_embedding': idx}]
    The 'clip_embedding' field is the integer index into the top-level embeddings list,
    which is exactly what train.py expects.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(clip_model_type, device=device)
    model.eval()

    embeddings: List[np.ndarray] = []
    captions: List[dict] = []

    idx = 0
    for (img_name, text) in tqdm(pairs, ncols=100):
        img_path = os.path.join(images_root, img_name)
        if not os.path.exists(img_path):
            # skip missing files silently; you can log if you prefer
            continue
        try:
            image = load_padchest_image(img_path)
        except Exception:
            # unreadable file -> skip
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)  # (1, D)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        vec = image_features[0].detach().cpu().numpy().astype(np.float32)  # (D,)
        embeddings.append(vec)

        # You can store image_name as image_id if you want; the code only uses it as an identifier.
        captions.append({
            "image_id": img_name,          # keep the actual filename
            "caption": text or "",         # empty string if no caption
            "clip_embedding": idx          # integer index into 'embeddings'
        })
        idx += 1

    return embeddings, captions


def save_pkl(embs: List[np.ndarray], caps: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    obj = {
        "clip_embedding": embs,
        "captions": caps,
    }
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {len(caps)} samples -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=str, default="data/padchestgr/PadChest_GR",
                    help="Folder with all PadChest_GR images")
    ap.add_argument("--train_csv", type=str, default="data/padchestgr/train_final_separated.csv")
    ap.add_argument("--test_csv",  type=str, default="data/padchestgr/test_final_separated.csv")
    ap.add_argument("--out_train", type=str, default="data/padchestgr/padchestgr_clip_ViT-B_32_train_PADCHESTGR_INDEX.pkl")
    ap.add_argument("--out_test",  type=str, default="data/padchestgr/padchestgr_clip_ViT-B_32_test_PADCHESTGR_INDEX.pkl")
    ap.add_argument("--clip_model_type", type=str, default="ViT-B/32",
                    choices=["ViT-B/32", "ViT-B/16", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"])
    ap.add_argument("--no_normalize", action="store_true", help="Disable L2-normalization of CLIP features")
    args = ap.parse_args()

    normalize = not args.no_normalize

    # ---------- TRAIN ----------
    train_pairs = read_csv_pairs(args.train_csv)
    # Optionally drop empty captions for training:
    train_pairs = [(i, c) for (i, c) in train_pairs if c and c.strip()]

    train_embs, train_caps = encode_images_with_clip(
        train_pairs, args.images_root, clip_model_type=args.clip_model_type, normalize=normalize
    )
    save_pkl(train_embs, train_caps, args.out_train)

    # ---------- TEST ----------
    test_pairs = read_csv_pairs(args.test_csv)  # may contain empty captions; keep as-is
    test_embs, test_caps = encode_images_with_clip(
        test_pairs, args.images_root, clip_model_type=args.clip_model_type, normalize=normalize
    )
    save_pkl(test_embs, test_caps, args.out_test)


if __name__ == "__main__":
    main()
