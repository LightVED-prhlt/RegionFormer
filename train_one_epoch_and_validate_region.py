# train_one_epoch_and_validate.py
# One-epoch training, validation (loss + perplexity), and CSV with columns: imgid, ref, hyp

import os
import sys
import json
import math
import pickle
import argparse
from typing import Optional, List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from RegionFormer.RegionFormer import ClipCaptionModel, ClipCaptionPrefix, MappingType

CPU = torch.device("cpu")

# ----------------------------
# Config
# ----------------------------

def save_config(args: argparse.Namespace):
    cfg = {k: v for k, v in args._get_kwargs()}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.prefix}.json"), "w") as f:
        json.dump(cfg, f, indent=2)

# ----------------------------
# Dataset
# ----------------------------
class ClipCocoDataset(Dataset):
    """
    Expects PKL with:
      - "clip_embedding": [N_embeddings, D] array/tensor (this pool contains BOTH image and mask embeddings)
      - "captions": list of dicts:
          {
            "image_id": str,
            "caption": str,
            "clip_embedding": int,            # index of the image embedding in the pool
            "clip_mask_embedding": int | None # (optional) index of the mask embedding in the pool
          }

    If 'clip_mask_embedding' is present and --use_mask is enabled, each sample returns
    the concatenation [image_emb, mask_emb] as the prefix input vector. Otherwise only image_emb.
    """
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix: bool = False, use_mask: bool = False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_mask = use_mask

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        print(f"Embedding pool size: {len(all_data['clip_embedding'])}")
        sys.stdout.flush()

        self.prefixes = (torch.tensor(all_data["clip_embedding"])
                         if not torch.is_tensor(all_data["clip_embedding"])
                         else all_data["clip_embedding"])
        captions_raw = all_data["captions"]

        # Determine if this PKL actually has mask embeddings
        self.has_mask: bool = len(captions_raw) > 0 and ("clip_mask_embedding" in captions_raw[0])

        self.image_ids: List[str] = [c["image_id"] for c in captions_raw]
        self.captions: List[str] = [c["caption"] for c in captions_raw]

        # Build indices (image and maybe mask)
        self.caption2embedding: List[int] = [c["clip_embedding"] for c in captions_raw]
        self.caption2maskembedding: Optional[List[int]] = None
        if self.has_mask:
            # could be None / missing for some rows; use -1 as sentinel
            tmp = []
            for c in captions_raw:
                v = c.get("clip_mask_embedding", None)
                tmp.append(int(v) if v is not None else -1)
            self.caption2maskembedding = tmp

        # Token cache keyed by mask usage to avoid shape mismatches across runs
        mask_suffix = "_mask" if self.has_mask and self.use_mask else ""
        token_cache = f"{data_path[:-4]}_tokens{mask_suffix}.pkl"

        if os.path.isfile(token_cache):
            try:
                with open(token_cache, 'rb') as f:
                    payload = pickle.load(f)
                # old caches had 3 items; new caches have 4 (include mask mapping)
                if isinstance(payload, list) and (len(payload) in (3,4)):
                    if len(payload) == 3:
                        self.captions_tokens, self.caption2embedding, max_seq_len = payload
                        # no mask indices in cache; keep what we already built from PKL
                    else:
                        self.captions_tokens, self.caption2embedding, self.caption2maskembedding, max_seq_len = payload
                else:
                    raise ValueError("Bad token cache format; regenerating.")
            except Exception:
                self._build_tokens_and_cache(token_cache)
        else:
            self._build_tokens_and_cache(token_cache)

        all_len = torch.tensor([len(t) for t in self.captions_tokens]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def _build_tokens_and_cache(self, token_cache: str):
        self.captions_tokens = []
        max_seq_len = 0
        for c in self.captions:
            toks = self.tokenizer.encode(c)
            self.captions_tokens.append(torch.tensor(toks, dtype=torch.int64))
            max_seq_len = max(max_seq_len, len(toks))
        os.makedirs(os.path.dirname(token_cache), exist_ok=True) if os.path.dirname(token_cache) else None
        # Save both mappings when available
        if self.has_mask and self.use_mask:
            with open(token_cache, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, self.caption2maskembedding, max_seq_len], f)
        else:
            with open(token_cache, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        # prefix mask (prefix_length ones) + token mask
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)

        # Get image embedding
        idx_img = int(self.caption2embedding[item])
        prefix_img = self.prefixes[idx_img]

        # Optionally get mask embedding and concat
        use_mask_here = self.use_mask and self.has_mask and (self.caption2maskembedding is not None) \
                        and (self.caption2maskembedding[item] is not None) and (self.caption2maskembedding[item] >= 0)
        if use_mask_here:
            idx_mask = int(self.caption2maskembedding[item])
            prefix_mask = self.prefixes[idx_mask]
            if self.normalize_prefix:
                prefix_img = prefix_img.float()
                prefix_img = prefix_img / (prefix_img.norm(2, -1) + 1e-8)
                prefix_mask = prefix_mask.float()
                prefix_mask = prefix_mask / (prefix_mask.norm(2, -1) + 1e-8)
            prefix = torch.cat([prefix_img, prefix_mask], dim=-1)
        else:
            prefix = prefix_img
            if self.normalize_prefix:
                prefix = prefix.float()
                prefix = prefix / (prefix.norm(2, -1) + 1e-8)

        image_id = self.image_ids[item]
        ref = self.captions[item]
        return tokens, mask, prefix, image_id, ref

# ----------------------------
# Text generation (beam / nucleus)
# ----------------------------
@torch.no_grad()
def generate_beam(model, tokenizer, embed, beam_size: int = 5, entry_length: int = 67,
                  temperature: float = 1.0, stop_token: str = ".") -> str:
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    generated = embed
    for _ in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()
        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            tokens = next_tokens
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = (next_tokens % scores_sum.shape[1]).unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]
        next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        is_stopped = is_stopped | next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[: int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    return output_texts[order[0]]

@torch.no_grad()
def generate_nucleus(model, tokenizer, embed, entry_length: int = 67, top_p: float = 0.8,
                     temperature: float = 1.0, stop_token: str = ".") -> str:
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")

    local_tokens = None
    generated = embed
    for _ in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.gpt.transformer.wte(next_token)
        local_tokens = next_token if local_tokens is None else torch.cat((local_tokens, next_token), dim=1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        if stop_token_index == next_token.item():
            break

    output_list = list(local_tokens.squeeze().cpu().numpy())
    return GPT2Tokenizer.from_pretrained("gpt2").decode(output_list)

# ----------------------------
# Train one epoch
# ----------------------------
def train_one_epoch(dataset: ClipCocoDataset, model: nn.Module, args,
                    optimizer: torch.optim.Optimizer, scheduler) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()
    if isinstance(model, ClipCaptionPrefix):
        model.gpt.eval()

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=True)

    progress = tqdm(total=len(loader), desc="train(epoch)")
    for tokens, mask, prefix, _, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        tokens = tokens.to(device)
        mask = mask.to(device)
        prefix = prefix.to(device, dtype=torch.float32)

        outputs = model(tokens, prefix, mask)
        logits = outputs.logits[:, dataset.prefix_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tokens.flatten(), ignore_index=0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
        progress.update(1)
    progress.close()

# ----------------------------
# Validation: loss + CSV
# ----------------------------
@torch.no_grad()
def validate_loss(val_ds: ClipCocoDataset, model: nn.Module) -> float:
    """
    Token-level average CE loss with ignore_index=0 (padding).
    Uses reduction='sum' to get true token-avg over the entire set.
    """
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    total_loss_sum = 0.0
    total_tokens = 0

    for tokens, mask, prefix, _, _ in tqdm(loader, desc="val/loss"):
        tokens = tokens.to(device)
        mask = mask.to(device)
        prefix = prefix.to(device, dtype=torch.float32)

        outputs = model(tokens, prefix, mask)
        logits = outputs.logits[:, val_ds.prefix_length - 1: -1]  # align with targets
        # sum over all non-pad tokens
        loss_sum = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     tokens.flatten(), ignore_index=0, reduction='sum')
        total_loss_sum += loss_sum.item()
        total_tokens += (tokens != 0).sum().item()

    avg_loss = total_loss_sum / max(1, total_tokens)
    return avg_loss

@torch.no_grad()
def validate_and_save_csv(val_ds: ClipCocoDataset, model: nn.Module, args) -> str:
    device = next(model.parameters()).device
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    rows = []
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    for tokens, mask, prefix, image_id, ref in tqdm(loader, desc="val/generate"):
        prefix = prefix.to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

        if args.beam:
            hyp = generate_beam(model, tokenizer, embed=prefix_embed,
                                beam_size=args.beam_size, entry_length=args.gen_len,
                                temperature=args.temp, stop_token=".")
        else:
            hyp = generate_nucleus(model, tokenizer, embed=prefix_embed,
                                   entry_length=args.gen_len, top_p=args.top_p,
                                   temperature=args.temp, stop_token=".")
        rows.append({"imgid": image_id[0], "ref": ref[0], "hyp": hyp})

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.prefix}_val_captions.csv")
    pd.DataFrame(rows, columns=["imgid", "ref", "hyp"]).to_csv(csv_path, index=False)
    return csv_path

# ----------------------------
# Orchestration
# ----------------------------
def build_model(args):
    # Base CLIP embedding size (ViT/RN)
    base_dim = 640 if args.is_rn else 512
    # If using mask embeddings, we concatenated [img, mask] -> double dim
    prefix_dim = base_dim * 2 if args.use_mask else base_dim
    map_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length,
                                  clip_length=args.prefix_length_clip,
                                  prefix_size=prefix_dim,
                                  num_layers=args.num_layers,
                                  mapping_type=map_type)
    else:
        model = ClipCaptionModel(args.prefix_length,
                                 clip_length=args.prefix_length_clip,
                                 prefix_size=prefix_dim,
                                 num_layers=args.num_layers,
                                 mapping_type=map_type)
    return model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

def parse_args():
    p = argparse.ArgumentParser(description="One-epoch training + validation (loss & CSV)")
    # Data
    p.add_argument('--train_pkl', type=str, required=True)
    p.add_argument('--val_pkl',   type=str, required=True)
    p.add_argument('--out_dir',   type=str, required=True)
    p.add_argument('--prefix',    type=str, default='coco_prefix')

    # Model / mapper
    p.add_argument('--only_prefix', action='store_true')
    p.add_argument('--mapping_type', type=str, default='mlp', choices=['mlp','transformer'])
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--prefix_length', type=int, default=10)
    p.add_argument('--prefix_length_clip', type=int, default=10)
    p.add_argument('--is_rn', action='store_true')
    p.add_argument('--normalize_prefix', action='store_true')
    p.add_argument('--use_mask', action='store_true', help="Concat CLIP mask embedding with image embedding")

    # Train
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--bs', type=int, default=40)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup_steps', type=int, default=500)

    # Generation
    p.add_argument('--beam', action='store_true')
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--gen_len', type=int, default=67)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--temp', type=float, default=1.0)

    return p.parse_args()

def main():
    args = parse_args()
    save_config(args)

    # Datasets (pass use_mask to control concatenation behavior)
    train_ds = ClipCocoDataset(args.train_pkl, args.prefix_length,
                               normalize_prefix=args.normalize_prefix, use_mask=args.use_mask)
    val_ds   = ClipCocoDataset(args.val_pkl, args.prefix_length,
                               normalize_prefix=args.normalize_prefix, use_mask=args.use_mask)

    # Model
    model = build_model(args)

    # Optimizer & scheduler created ONCE for all epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # steps per epoch (drop_last=True like in the DataLoader inside train_one_epoch)
    steps_per_epoch = len(train_ds) // args.bs
    total_train_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_train_steps
    )

    best_val_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs}")
        train_one_epoch(train_ds, model, args, optimizer, scheduler)

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.out_dir, f"{args.prefix}-{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Validation each epoch
        print(">>> Computing validation loss")
        val_loss = validate_loss(val_ds, model)
        val_ppl = math.exp(min(50, val_loss))
        print(f"[Validation][epoch {epoch}] loss={val_loss:.6f}  ppl={val_ppl:.3f}")

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.out_dir, f"{args.prefix}-best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best; saved: {best_path}")

        # Save per-epoch metrics
        metrics_path = os.path.join(args.out_dir, f"{args.prefix}_val_metrics_epoch_{epoch:03d}.json")
        with open(metrics_path, "w") as f:
            json.dump({"epoch": epoch, "val_loss": val_loss, "val_ppl": val_ppl}, f, indent=2)
        print(f"Wrote metrics: {metrics_path}")

    # (Optional) Generate CSV once at the end with the best model
    print(">>> Loading best checkpoint and generating validation captions CSV")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f"{args.prefix}-best.pt"),
                                     map_location=next(model.parameters()).device))
    csv_path = validate_and_save_csv(val_ds, model, args)
    print(f"Wrote CSV: {csv_path}")

if __name__ == "__main__":
    main()
