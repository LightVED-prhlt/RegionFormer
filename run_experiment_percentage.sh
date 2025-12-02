#!/bin/bash

# Base arguments
TRAIN_PKL=./data/padchestgr/padchestgr_clip_ViT-B_32_train_PADCHESTGR_REGION_INDEX.pkl
VAL_PKL=./data/padchestgr/padchestgr_clip_ViT-B_32_test_PADCHESTGR_REGION_INDEX.pkl
PREFIX=padchestgr_prefix

# Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_5/


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
  --use_mask \
  --percent 5


# Run WITH mask only
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_only_5/


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
  --mask_only \
  --percent 5


# Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_10/


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
  --use_mask \
  --percent 10


# Run WITH mask only
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_only_10/


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
  --mask_only \
  --percent 10


# Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_20/


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
  --use_mask \
  --percent 20


# Run WITH mask only
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_only_20/


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
  --mask_only \
  --percent 20


# Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_50/


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
  --use_mask \
  --percent 50


# Run WITH mask only
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_only_50/


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
  --mask_only \
  --percent 50

  # Run WITH mask
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_80/


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
  --use_mask \
  --percent 80


# Run WITH mask only
OUT_DIR=./EXPERIMENTS/padchestgr_train_tf_prefix_jpg_region_15_with_mask_only_80/


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
  --mask_only \
  --percent 80