#!/bin/bash

# Base arguments
TRAIN_PKL=./data/padchestgr/padchestgr_clip_ViT-B_32_train_PADCHESTGR_REGION_INDEX.pkl
VAL_PKL=./data/padchestgr/padchestgr_clip_ViT-B_32_test_PADCHESTGR_REGION_INDEX.pkl
PREFIX=padchestgr_prefix

# Run WITHOUT mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_without_mask/

echo "Running experiment WITHOUT mask..."
python train_one_epoch_and_validate_region.py \
  --train_pkl "$TRAIN_PKL" \
  --val_pkl "$VAL_PKL" \
  --out_dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --only_prefix \
  --mapping_type transformer \
  --num_layers 8 \
  --prefix_length 40 \
  --prefix_length_clip 40 \
  --bs 40 \
  --beam \
  --gen_len 15 \
  --epochs 20


# Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask/


echo "Running experiment WITH mask..."
python train_one_epoch_and_validate_region.py \
  --train_pkl "$TRAIN_PKL" \
  --val_pkl "$VAL_PKL" \
  --out_dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --only_prefix \
  --mapping_type transformer \
  --num_layers 8 \
  --prefix_length 40 \
  --prefix_length_clip 40 \
  --bs 40 \
  --beam \
  --gen_len 15 \
  --epochs 20 \
  --use_mask


# Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_only/


echo "Running experiment WITH mask only..."
python train_one_epoch_and_validate_region.py \
  --train_pkl "$TRAIN_PKL" \
  --val_pkl "$VAL_PKL" \
  --out_dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --only_prefix \
  --mapping_type transformer \
  --num_layers 8 \
  --prefix_length 40 \
  --prefix_length_clip 40 \
  --bs 40 \
  --beam \
  --gen_len 15 \
  --epochs 20 \
  --mask_only
