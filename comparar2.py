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
# Helpers
# ======================================================
def truncate_words(text: str, max_words: int = 20) -> str:
    return " ".join(str(text).split()[:max_words])

def norm_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def norm_imgid(imgid: str) -> str:
    return norm_text(imgid.replace(".", ""))

def safe_outname(imgid: str, ref: str) -> str:
    return f"{norm_imgid(imgid)}_{norm_text(truncate_words(ref))}.jpg"

# ======================================================
# PNG-SAFE medical image loader
# ======================================================
def load_image(path: Path) -> Image.Image:
    """
    Correctly loads PadChest PNGs:
    - Handles 16-bit grayscale
    - Handles RGBA
    - Applies robust contrast normalization
    """
    im = Image.open(path)

    # RGBA → RGB
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        return im

    # 16-bit or float grayscale
    if im.mode in ("I;16", "I", "F"):
        arr = np.array(im).astype(np.float32)

        vmin = np.percentile(arr, 1)
        vmax = np.percentile(arr, 99)
        if vmax <= vmin:
            vmin, vmax = arr.min(), arr.max() if arr.max() > arr.min() else (0.0, 1.0)

        arr = np.clip(arr, vmin, vmax)
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
        arr = (arr * 255.0).astype(np.uint8)

        im8 = Image.fromarray(arr, mode="L")
        im8 = ImageOps.autocontrast(im8)
        return im8.convert("RGB")

    # Standard grayscale
    if im.mode == "L":
        im = ImageOps.autocontrast(im)
        return im.convert("RGB")

    # Already OK
    return im.convert("RGB")

# ======================================================
# Draw bounding boxes (normalized coords)
# ======================================================
def draw_boxes(image: Image.Image, boxes, color="red", width=3):
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
    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.18, wspace=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ======================================================
# Load CSVs
# ======================================================
df_test = pd.read_csv(csv_test)
df_wo   = pd.read_csv(csv_without)
df_wi   = pd.read_csv(csv_with)

assert len(df_test) == len(df_wo) == len(df_wi)

# ======================================================
# Load attention images (ORDERED)
# ======================================================
imgs_wo = sorted(list(imgdir_without.glob("*.jpg")) + list(imgdir_without.glob("*.png")))
imgs_wi = sorted(list(imgdir_with.glob("*.jpg")) + list(imgdir_with.glob("*.png")))

assert len(imgs_wo) == len(imgs_wi) == len(df_test)

# ======================================================
# Main loop
# ======================================================
created = 0

for i in range(len(df_test)):
    row_test = df_test.iloc[i]
    row_wo   = df_wo.iloc[i]
    row_wi   = df_wi.iloc[i]

    img_name = row_test["ImageID"]
    ref_txt  = row_test["report_en"]
    hyp_wo   = row_wo["hyp"]
    hyp_wi   = row_wi["hyp"]

    try:
        orig_img = load_image(orig_imgdir / img_name)
        boxes = ast.literal_eval(row_test["boxes"])
        orig_img = draw_boxes(orig_img, boxes)

        im_wo = load_image(imgs_wo[i])
        im_wi = load_image(imgs_wi[i])

        out_path = outdir / safe_outname(img_name, ref_txt)

        make_figure(
            orig_img, im_wo, im_wi,
            ref_txt, hyp_wo, hyp_wi,
            out_path
        )

        created += 1

    except Exception as e:
        print(f"[ERROR] index={i}, image={img_name}: {e}")

print(f"✅ Done. Created {created} 3-panel figures.")
print(f"📁 Output directory: {outdir}")
