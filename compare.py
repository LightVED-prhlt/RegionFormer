import os
from pathlib import Path
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from textwrap import fill

# --------------------
# Paths
# --------------------
csv_without = Path("/home/moha/Desktop/RegionFormer/EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_without_mask_img_cr/padchestgr_prefix_val_captions.csv")
csv_with    = Path("/home/moha/Desktop/RegionFormer/EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_img_cr/padchestgr_prefix_val_captions.csv")

imgdir_without = Path("/home/moha/Desktop/RegionFormer/EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_without_mask_img_cr/attn_jpg_legrad")
imgdir_with    = Path("/home/moha/Desktop/RegionFormer/EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_img_cr/attn_jpg_legrad")

outdir = Path("/home/moha/Desktop/RegionFormer/comparison")
outdir.mkdir(parents=True, exist_ok=True)

# --------------------
# Helpers
# --------------------
def truncate_words(text: str, max_words: int = 20) -> str:
    words = str(text).split()
    return " ".join(words[:max_words])

def norm_imgid(imgid: str) -> str:
    # remove dots, keep underscores/alnum
    imgid = str(imgid).strip().replace(".", "")
    imgid = re.sub(r"[^0-9A-Za-z_]", "_", imgid)
    imgid = re.sub(r"_+", "_", imgid).strip("_")
    return imgid

def norm_text(s: str) -> str:
    # spaces -> _, non-alnum -> _, collapse _
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def build_prefix(imgid: str, ref: str) -> str:
    """Prefix used to find files: {imgid}_{ref}_ (no hyp)."""
    return f"{norm_imgid(imgid)}_{norm_text(ref)}_"

def find_by_prefix(directory: Path, prefix: str):
    """Find the first file whose name starts with prefix (prefer .jpg, fallback .png).
       If multiple, pick the shortest (safest)."""
    # Collect both jpg and png
    candidates = list(directory.glob(prefix + "*.jpg")) + list(directory.glob(prefix + "*.png"))
    if not candidates:
        return None
    # Prefer jpg, then shortest path
    jpgs = [p for p in candidates if p.suffix.lower() == ".jpg"]
    pool = jpgs if jpgs else candidates
    pool.sort(key=lambda p: len(str(p)))
    return pool[0]

def load_image(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB")

def safe_ref_outname(imgid: str, ref: str, max_words: int = 20, max_len: int = 240) -> str:
    """Save using imgid_Reftext (<=20 words), sanitized and length-limited."""
    ref_trunc = truncate_words(ref, max_words)
    base_ref = norm_text(ref_trunc)
    if len(base_ref) > max_len:
        base_ref = base_ref[:max_len]
    base_imgid = norm_imgid(imgid)
    return f"{base_imgid}_{base_ref}.jpg"


def make_figure(img_left, img_right, ref_text, hyp_left, hyp_right, save_path: Path):
    plt.figure(figsize=(12, 8))

    # Left: WITHOUT MASK
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_left)
    ax1.axis("off")
    ax1.set_title("Without mask", fontsize=11, pad=8)
    ax1.text(0.5, -0.10, fill(hyp_left, width=60),
             transform=ax1.transAxes, ha="center", va="top", fontsize=9)

    # Right: WITH MASK
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img_right)
    ax2.axis("off")
    ax2.set_title("With mask", fontsize=11, pad=8)
    ax2.text(0.5, -0.10, fill(hyp_right, width=60),
             transform=ax2.transAxes, ha="center", va="top", fontsize=9)

    # Common ref at top (full ref shown in figure title; only truncated in filename)
    plt.suptitle(fill(ref_text, width=100), fontsize=13, y=0.98)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.14, wspace=0.05)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

# --------------------
# Load & merge CSVs
# --------------------
df_wo = pd.read_csv(csv_without)
df_wi = pd.read_csv(csv_with)

df_wo.rename(columns={c: c.strip().lower() for c in df_wo.columns}, inplace=True)
df_wi.rename(columns={c: c.strip().lower() for c in df_wi.columns}, inplace=True)

df = df_wo.merge(df_wi, on=["imgid", "ref"], suffixes=("_without", "_with"))
print(f"Merged {len(df)} rows.")

# --------------------
# Process
# --------------------
missing = 0
created = 0

for _, row in df.iterrows():
    imgid   = str(row["imgid"])
    ref_txt = str(row["ref"])
    hyp_wo  = str(row["hyp_without"])
    hyp_wi  = str(row["hyp_with"])

    # Find files by prefix {imgid}_{ref}_ (ignore hyp in lookup to avoid long path issues)
    prefix = build_prefix(imgid, ref_txt)
    path_wo = find_by_prefix(imgdir_without, prefix)
    path_wi = find_by_prefix(imgdir_with,    prefix)

    if not path_wo or not path_wi:
        print(f"[WARN] Missing image(s) for imgid={imgid} "
              f"{'(without mask)' if not path_wo else ''} "
              f"{'(with mask)' if not path_wi else ''}")
        missing += 1
        continue

    try:
        im_left  = load_image(path_wo)
        im_right = load_image(path_wi)
        out_name = safe_ref_outname(imgid, ref_txt)  # save using ONLY the ref (<=20 words)
        make_figure(im_left, im_right, ref_txt, hyp_wo, hyp_wi, outdir / out_name)
        created += 1
    except Exception as e:
        print(f"[ERROR] {imgid}: {e}")

print(f"✅ Done. Created: {created}, Missing pairs skipped: {missing}. Output: {outdir}")
