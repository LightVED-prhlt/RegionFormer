import json
import csv
import ast
from collections import defaultdict

# -------------------------
# File paths
# -------------------------
JSON_PATH = "grounded_reports_20240819.json"
MASTER_CSV_PATH = "master_table.csv"

OUTPUT_FILES = {
    "train": "train_final_separated.csv",
    "test": "test_final_separated.csv",
    "validation": "validation_final_separated.csv",
}

# -------------------------
# Load grounded reports
# -------------------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    grounded_reports = json.load(f)

reports_by_study = {
    r["StudyID"]: r for r in grounded_reports
}

# -------------------------
# Load master table (split info)
# -------------------------
study_split = {}
study_image = {}

with open(MASTER_CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row["StudyID"]
        study_split[sid] = row["split"]
        study_image[sid] = row["ImageID"]

# -------------------------
# Prepare output containers
# -------------------------
rows_by_split = defaultdict(list)

# -------------------------
# Process studies
# -------------------------
for study_id, report in reports_by_study.items():

    if study_id not in study_split:
        continue  # study not in master_table

    split = study_split[study_id]
    image_id = study_image.get(study_id, report.get("ImageID"))

    for idx, finding in enumerate(report.get("findings", [])):

        sentence_en = finding.get("sentence_en", "")
        sentence_es = finding.get("sentence_es", "")

        boxes = finding.get("boxes", [])
        extra_boxes = finding.get("extra_boxes", [])

        # Format exactly as requested
        boxes_formatted = [[
            0,
            sentence_en,
            boxes
        ]]

        extra_boxes_formatted = [[
            0,
            sentence_en,
            extra_boxes
        ]]

        row = {
            "StudyID": study_id,
            "ImageID": image_id,
            "report_es": sentence_es,
            "report_en": sentence_en,
            "boxes": json.dumps(boxes_formatted, ensure_ascii=False),
            "extra_boxes": json.dumps(extra_boxes_formatted, ensure_ascii=False),
            "split": split,
        }

        rows_by_split[split].append(row)

# -------------------------
# Write CSVs
# -------------------------
FIELDNAMES = [
    "StudyID",
    "ImageID",
    "report_es",
    "report_en",
    "boxes",
    "extra_boxes",
    "split",
]

for split, out_path in OUTPUT_FILES.items():

    if split not in rows_by_split:
        continue

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_by_split[split])

    print(f"✔ Saved {out_path} ({len(rows_by_split[split])} rows)")
