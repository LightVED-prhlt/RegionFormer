import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from textwrap import fill

# ======================================================
# Paths
# ======================================================
csv_test = Path(
    "/home/moha/Desktop/RegionFormer/data/padchestgr/test_final_separated.csv"
)

csv_without = Path(
    "/home/moha/Desktop/RegionFormer/EXPERIMENTS/"
    "padchestgr_train_tf_prefix_jpg_region_15_without_mask_img_cr/"
    "padchestgr_prefix_val_captions.csv"
)

csv_with = Path(
    "/home/moha/Desktop/RegionFormer/EXPERIMENTS/"
    "padchestgr_train_tf_prefix_jpg_region_15_with_mask_img_cr/"
    "padchestgr_prefix_val_captions.csv"
)

imgdir_without = Path(
    "/home/moha/Desktop/RegionFormer/EXPERIMENTS/"
    "padchestgr_train_tf_prefix_jpg_region_15_without_mask_img_cr/"
    "attn_jpg_legrad"
)

imgdir_with = Path(
    "/home/moha/Desktop/RegionFormer/EXPERIMENTS/"
    "padchestgr_train_tf_prefix_jpg_region_15_with_mask_img_cr/"
    "attn_jpg_legrad"
)

orig_imgdir = Path(
    "/home/moha/Desktop/RegionFormer/data/padchestgr/PadChest_GR"
)

outdir = Path("/home/moha/Desktop/RegionFormer/comparison_3panel")
outdir.mkdir(parents=True, exist_ok=True)

# ======================================================
# Text helpers
# ======================================================
def norm_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def truncate_words(text, max_words=20):
    return " ".join(str(text).split()[:max_words])

def build_prefix(image_id: str, report_en: str) -> str:
    img_part = image_id.replace(".png", "png")
    rep_part = norm_text(report_en)
    return f"{img_part}_{rep_part}_"

def safe_outname(imgid, ref):
    return f"{norm_text(imgid)}_{norm_text(truncate_words(ref))}.jpg"

# ======================================================
# Robust medical PNG loader
# ======================================================
def load_image(path: Path) -> Image.Image:
    im = Image.open(path)

    if im.mode in ("I;16", "I", "F"):
        arr = np.array(im).astype(np.float32)
        vmin = np.percentile(arr, 1)
        vmax = np.percentile(arr, 99)
        arr = np.clip(arr, vmin, vmax)
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        im = Image.fromarray(arr)
        im = ImageOps.autocontrast(im)
        return im.convert("RGB")

    if im.mode == "L":
        return ImageOps.autocontrast(im).convert("RGB")

    if im.mode == "RGBA":
        return im.convert("RGB")

    return im.convert("RGB")

# ======================================================
# Draw GT boxes
# ======================================================
def draw_boxes(image, boxes, color="red", width=5):
    draw = ImageDraw.Draw(image)
    W, H = image.size
    for entry in boxes:
        _, _, box_list = entry
        for x1, y1, x2, y2 in box_list:
            draw.rectangle(
                [x1 * W, y1 * H, x2 * W, y2 * H],
                outline=color,
                width=width
            )
    return image

# ======================================================
# Find attention image by prefix
# ======================================================
def find_by_prefix(directory: Path, prefix: str):
    matches = list(directory.glob(prefix + "*.jpg"))
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.name))
    return matches[0]

# ======================================================
# Figure (3 panels)
# ======================================================
def make_figure(img_orig, img_wo, img_wi,
                ref_text, hyp_wo, hyp_wi, save_path):

    plt.figure(figsize=(18, 7))

    ax0 = plt.subplot(1, 3, 1)
    ax0.imshow(img_orig)
    ax0.axis("off")
    ax0.set_title("Original + GT boxes", fontsize=11)

    ax1 = plt.subplot(1, 3, 2)
    ax1.imshow(img_wo)
    ax1.axis("off")
    ax1.set_title("Without mask", fontsize=11)
    ax1.text(0.5, -0.12, fill(hyp_wo, 55),
             transform=ax1.transAxes, ha="center", va="top", fontsize=9)

    ax2 = plt.subplot(1, 3, 3)
    ax2.imshow(img_wi)
    ax2.axis("off")
    ax2.set_title("With mask", fontsize=11)
    ax2.text(0.5, -0.12, fill(hyp_wi, 55),
             transform=ax2.transAxes, ha="center", va="top", fontsize=9)

    plt.suptitle(fill(ref_text, 110), fontsize=13, y=0.98)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.18)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ======================================================
# Load CSVs
# ======================================================
df_test = pd.read_csv(csv_test)
df_wo = pd.read_csv(csv_without)
df_wi = pd.read_csv(csv_with)

df_wo.columns = [c.lower() for c in df_wo.columns]
df_wi.columns = [c.lower() for c in df_wi.columns]

# ======================================================
# Main loop (PREFIX-BASED, CORRECT)
# ======================================================
created = 0
missing = 0

for _, row in df_test.iterrows():
    imgid = row["ImageID"]
    ref_txt = row["report_en"]

    prefix = build_prefix(imgid, ref_txt)

    attn_wo = find_by_prefix(imgdir_without, prefix)
    attn_wi = find_by_prefix(imgdir_with, prefix)

    if attn_wo is None or attn_wi is None:
        missing += 1
        continue

    row_wo = df_wo[df_wo["imgid"] == imgid]
    row_wi = df_wi[df_wi["imgid"] == imgid]

    if row_wo.empty or row_wi.empty:
        missing += 1
        continue

    try:
        orig_img = load_image(orig_imgdir / imgid)
        boxes = ast.literal_eval(row["boxes"])
        orig_img = draw_boxes(orig_img, boxes)

        im_wo = load_image(attn_wo)
        im_wi = load_image(attn_wi)

        out_path = outdir / safe_outname(imgid, ref_txt)

        make_figure(
            orig_img,
            im_wo,
            im_wi,
            ref_txt,
            row_wo.iloc[0]["hyp"],
            row_wi.iloc[0]["hyp"],
            out_path
        )

        created += 1

    except Exception as e:
        print(f"[ERROR] {imgid}: {e}")

print(f"✅ Done. Created: {created}, Missing: {missing}")
print(f"📁 Output directory: {outdir}")
