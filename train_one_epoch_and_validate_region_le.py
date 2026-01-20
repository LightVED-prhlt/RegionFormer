# One-epoch training, validation (loss + perplexity), captions CSV,
# and LEGRAD EXPLAINABILITY HEATMAPS overlaid on images.

import os
import sys
import re
import json
import math
import pickle
import argparse
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from RegionFormer.RegionFormer_le import ClipCaptionModel, ClipCaptionPrefix, MappingType
from utils.padchest_utils import load_padchest_image, resolve_image_path, ImageResolver

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
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix: bool = False, use_mask: bool = False, mask_only: bool = False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_mask = use_mask
        self.mask_only = mask_only

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        print(f"Embedding pool size: {len(all_data['clip_embedding'])}")
        sys.stdout.flush()

        emb = all_data["clip_embedding"]
        if torch.is_tensor(emb):
            self.prefixes = emb
        else:
            self.prefixes = torch.from_numpy(np.asarray(emb))

        captions_raw = all_data["captions"]
        self.base_dir: str = all_data.get("base_dir", "") or ""

        self.has_mask: bool = len(captions_raw) > 0 and ("clip_mask_embedding" in captions_raw[0])
        self.image_ids: List[str] = [c["image_id"] for c in captions_raw]
        self.captions: List[str] = [c["caption"] for c in captions_raw]

        self.caption2embedding: List[int] = [c["clip_embedding"] for c in captions_raw]
        self.caption2maskembedding: Optional[List[int]] = None
        if self.has_mask:
            tmp = []
            for c in captions_raw:
                v = c.get("clip_mask_embedding", None)
                tmp.append(int(v) if v is not None else -1)
            self.caption2maskembedding = tmp

        if self.mask_only:
            mode_suffix = "_maskonly"
        elif self.use_mask:
            mode_suffix = "_imgmask"
        else:
            mode_suffix = "_img"
        token_cache = f"{data_path[:-4]}_tokens{mode_suffix}.pkl"

        if os.path.isfile(token_cache):
            try:
                with open(token_cache, 'rb') as f:
                    payload = pickle.load(f)
                if isinstance(payload, list) and (len(payload) in (3, 4)):
                    if len(payload) == 3:
                        self.captions_tokens, self.caption2embedding, max_seq_len = payload
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
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)
        idx_img = int(self.caption2embedding[item])
        prefix_img = self.prefixes[idx_img]

        has_valid_mask = (self.has_mask and self.caption2maskembedding is not None and
                          self.caption2maskembedding[item] is not None and self.caption2maskembedding[item] >= 0)

        if self.mask_only:
            if not has_valid_mask:
                raise RuntimeError("mask_only=True but sample has no mask embedding")
            idx_mask = int(self.caption2maskembedding[item])
            prefix_mask = self.prefixes[idx_mask]
            if self.normalize_prefix:
                prefix_mask = prefix_mask.float()
                prefix_mask = prefix_mask / (prefix_mask.norm(2, -1) + 1e-8)
            prefix = prefix_mask
        elif self.use_mask and has_valid_mask:
            idx_mask = int(self.caption2maskembedding[item])
            prefix_mask = self.prefixes[idx_mask]
            if self.normalize_prefix:
                prefix_img = prefix_img.float()
                prefix_img = prefix_img / (prefix_img.norm(2, -1) + 1e-8)
                prefix_mask = prefix_mask.float()
                prefix_mask = prefix_mask / (prefix_mask.norm(2, -1) + 1e-8)
            prefix = torch.cat([prefix_img, prefix_mask], dim=-1)
        else:
            if self.normalize_prefix:
                prefix_img = prefix_img.float()
                prefix_img = prefix_img / (prefix_img.norm(2, -1) + 1e-8)
            prefix = prefix_img

        image_id = self.image_ids[item]
        ref = self.captions[item]
        base_dir_str = self.base_dir
        return tokens, mask, prefix, image_id, ref, base_dir_str


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
    for tokens, mask, prefix, _, _, _ in loader:
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
# Validation helpers: LeGrad & Viz
# ----------------------------
def _nearest_grid(n: int) -> Tuple[int, int]:
    r = int(math.sqrt(n))
    while r > 0:
        if n % r == 0:
            return r, n // r
        r -= 1
    return 1, n


def _sanitize_for_filename(s: str, maxlen: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s[:maxlen] if len(s) > maxlen else s


def _legrad_to_grid(attn_layers_grads: List[torch.Tensor],
                    clip_length: int,
                    prefix_length: int) -> np.ndarray:
    """
    [LEGRAD] Implementation
    attn_layers_grads: List of gradients of the attention maps. Each is [B, N, M, H].
    LeGrad Formula: Average( ReLU(Gradient) ) over heads and layers.
    """
    # Use the gradient, not the raw weights
    with torch.no_grad():
        # 1. Stack gradients -> [L, B, N, M, H]
        #    N (query) = prefix_length + clip_length
        #    M (key)   = prefix_length + clip_length
        #    H (heads)
        grad_stack = torch.stack(attn_layers_grads, dim=0)

        # 2. Apply ReLU (LeGrad considers the gradient as the signal and removes negatives)
        #    Formula: (grad)^+
        grad_stack = torch.nn.functional.relu(grad_stack)

        # 3. Focus on Image contributions.
        #    We want to know: "Which image patches (keys 0..clip_len) influenced the prefix (queries)?"
        #    The "prediction" comes from the prefix tokens in the mapper.
        #    Query Slice: clip_length : clip_length + prefix_length
        #    Key Slice: 0 : clip_length
        q_slice = slice(clip_length, clip_length + prefix_length)
        k_slice = slice(0, clip_length)
        
        # Sliced: [L, B, prefix_len, clip_len, H]
        relevant_grads = grad_stack[:, :, q_slice, k_slice, :]

        # 4. Aggregate (Average) over:
        #    - Layers (dim 0)
        #    - Heads (dim -1)
        #    - Queries (dim 2, i.e., average influence over all prefix tokens)
        #    Result shape: [B, clip_len]
        aggregated = relevant_grads.mean(dim=(0, 2, 4))
        
        # 5. Normalize [0, 1] for visualization
        aggregated = aggregated - aggregated.min(dim=1, keepdim=True)[0]
        denom = aggregated.max(dim=1, keepdim=True)[0] + 1e-8
        aggregated = aggregated / denom

        heat_1d = aggregated[0].cpu().numpy()
        rows, cols = _nearest_grid(clip_length)
        heat_2d = heat_1d.reshape(rows, cols)
        return heat_2d


def _overlay_and_save(xray_arr: np.ndarray, heatmap2d: np.ndarray, out_path: str,
                      alpha: float = 0.35, cmap_name: str = "jet"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    H, W = xray_arr.shape
    base_rgb = np.stack([xray_arr, xray_arr, xray_arr], axis=-1)
    heatmap_img = Image.fromarray((heatmap2d * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    hm_np = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    heat_rgb = cm.get_cmap(cmap_name)(hm_np)[..., :3]
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blend = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    blend = (np.clip(blend, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(blend).save(out_path, quality=95, subsampling=0)


# ----------------------------
# Validation: LeGrad
# ----------------------------
@torch.no_grad()
def validate_loss(val_ds: ClipCocoDataset, model: nn.Module) -> float:
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    total_loss_sum = 0.0
    total_tokens = 0
    for tokens, mask, prefix, _, _, _ in tqdm(loader, desc="val/loss"):
        tokens = tokens.to(device)
        mask = mask.to(device)
        prefix = prefix.to(device, dtype=torch.float32)
        outputs = model(tokens, prefix, mask)
        logits = outputs.logits[:, val_ds.prefix_length - 1: -1]
        loss_sum = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     tokens.flatten(), ignore_index=0, reduction='sum')
        total_loss_sum += loss_sum.item()
        total_tokens += (tokens != 0).sum().item()
    return total_loss_sum / max(1, total_tokens)


# [LEGRAD] Main Function change
# We remove @torch.no_grad() because we need backprop for LeGrad.
def validate_and_save_csv_legrad(val_ds: ClipCocoDataset, model: nn.Module, args) -> str:
    device = next(model.parameters()).device
    model.eval()
    
    # Enable gradients for LeGrad computation
    torch.set_grad_enabled(True)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    rows = []
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    attn_dir = os.path.join(args.out_dir, "attn_jpg_legrad")
    os.makedirs(attn_dir, exist_ok=True)

    roots = [r.strip() for r in args.images_root.split(",") if r.strip()] if args.images_root else []
    pkl_dir = os.path.dirname(args.val_pkl)
    resolver = ImageResolver(roots=roots, extra_try=[pkl_dir, os.getcwd()])

    print(">>> Generating Captions and computing LeGrad maps...")

    for tokens_gt, mask_gt, prefix, image_id, ref, base_dir in tqdm(loader, desc="val/legrad"):
        prefix = prefix.to(device, dtype=torch.float32).requires_grad_(True)
        model.zero_grad()

        # 1. Forward Pass (Inference) to get the generated caption
        #    We do this in no_grad mode (conceptually) or just detach, 
        #    because we can't backprop through the beam search discrete choices easily.
        with torch.no_grad():
            proj_inference, _, _ = model.project_with_attn(prefix, return_attn=False)
            prefix_embed_inf = proj_inference.reshape(1, args.prefix_length, -1)
            
            if args.beam:
                hyp_text = generate_beam(model, tokenizer, embed=prefix_embed_inf,
                                         beam_size=args.beam_size, entry_length=args.gen_len,
                                         temperature=args.temp, stop_token=".")
            else:
                hyp_text = generate_nucleus(model, tokenizer, embed=prefix_embed_inf,
                                            entry_length=args.gen_len, top_p=args.top_p,
                                            temperature=args.temp, stop_token=".")

        rows.append({"imgid": image_id[0], "ref": ref[0], "hyp": hyp_text})

        # 2. LeGrad Computation
        #    Now we treat the generated caption as the target.
        #    We run a forward pass to align the gradients.
        
        # Tokenize the generated text to create targets
        hyp_tokens = tokenizer.encode(hyp_text)
        if len(hyp_tokens) == 0:
            continue
        
        hyp_tensor = torch.tensor([hyp_tokens], dtype=torch.int64, device=device)
        
        # We need to manually run the steps inside model.forward() to keep the graph 
        # connected to the attention maps returned by `project_with_attn`
        
        # Step A: Project prefix (TRACKING GRADIENTS)
        # Note: 'attns' here will have requires_grad=True and retain_grad() called in RegionFormer
        proj, attns, clip_len = model.project_with_attn(prefix, return_attn=True)
        prefix_projections = proj.view(-1, args.prefix_length, model.gpt_embedding_size)
        
        # Step B: Embed the hypothesis text
        embedding_text = model.gpt.transformer.wte(hyp_tensor)
        
        # Step C: Concatenate
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        # Step D: Run GPT forward
        # We want the probability of the *generated* tokens.
        # We pass the hypothesis as labels.
        # Construct labels: dummy for prefix + hypothesis tokens
        dummy_token = model.get_dummy_token(1, device)
        labels = torch.cat((dummy_token, hyp_tensor), dim=1)
        
        outputs = model.gpt(inputs_embeds=embedding_cat, labels=labels)
        
        # Step E: Compute Score (Negative Loss = Score)
        # We want to maximize the likelihood of the generated caption, so we backprop on the loss.
        # LeGrad relies on the gradient of the *score* (logit). 
        # Backpropagating the CrossEntropyLoss is equivalent to backpropagating -log(p).
        # We can just use the loss directly; the sign flip is handled by the direction of optimization,
        # but for explainability, we usually want gradient of the class score.
        # However, for generation, minimizing loss aligns the attention to what supports the sequence.
        loss = outputs.loss
        
        # Step F: Backward
        loss.backward()
        
        # Step G: LeGrad Extraction
        # Now attns[i].grad is populated
        if attns is not None and len(attns) > 0:
            try:
                # Collect gradients
                grads = [a.grad for a in attns if a.grad is not None]
                
                if len(grads) == len(attns):
                    # Compute Heatmap
                    heat2d = _legrad_to_grid(grads, clip_length=clip_len, prefix_length=args.prefix_length)

                    base_dir_str = base_dir[0] if isinstance(base_dir, (list, tuple)) else base_dir
                    img_path = resolver.resolve(image_id[0], base_dir=base_dir_str, pkl_dir=pkl_dir)
                    
                    if img_path:
                        xray_arr = load_padchest_image(img_path)
                        fname = f"{_sanitize_for_filename(os.path.basename(img_path))}_" \
                                f"{_sanitize_for_filename(ref[0])}_{_sanitize_for_filename(hyp_text)}.jpg"
                        out_path = os.path.join(attn_dir, fname)
                        _overlay_and_save(xray_arr, heat2d, out_path, alpha=getattr(args, "overlay_alpha", 0.35))
            except Exception as e:
                print(f"[warn] LeGrad viz failed for {image_id[0]}: {e}")
                import traceback
                traceback.print_exc()

    csv_path = os.path.join(args.out_dir, f"{args.prefix}_val_captions.csv")
    pd.DataFrame(rows, columns=["imgid", "ref", "hyp"]).to_csv(csv_path, index=False)
    
    # Restore safe default
    torch.set_grad_enabled(False)
    return csv_path


# ----------------------------
# Orchestration
# ----------------------------
def build_model(args):
    base_dim = 640 if args.is_rn else 512
    if args.use_mask and not args.mask_only:
        prefix_dim = base_dim * 2
    else:
        prefix_dim = base_dim

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
    p = argparse.ArgumentParser(description="One-epoch training + LeGrad Visualization")
    p.add_argument('--train_pkl', type=str, required=True)
    p.add_argument('--val_pkl',   type=str, required=True)
    p.add_argument('--out_dir',   type=str, required=True)
    p.add_argument('--prefix',    type=str, default='coco_prefix')

    p.add_argument('--only_prefix', action='store_true')
    p.add_argument('--mapping_type', type=str, default='mlp', choices=['mlp','transformer'])
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--prefix_length', type=int, default=10)
    p.add_argument('--prefix_length_clip', type=int, default=10)
    p.add_argument('--is_rn', action='store_true')
    p.add_argument('--normalize_prefix', action='store_true')
    p.add_argument('--use_mask', action='store_true')
    p.add_argument('--mask_only', action='store_true')

    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--bs', type=int, default=40)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup_steps', type=int, default=500)

    p.add_argument('--beam', action='store_true')
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--gen_len', type=int, default=67)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--temp', type=float, default=1.0)
    p.add_argument('--images_root', type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()
    if args.use_mask and args.mask_only:
        raise ValueError("choose either --use_mask or --mask_only, not both")
    save_config(args)

    train_ds = ClipCocoDataset(args.train_pkl, args.prefix_length,
                               normalize_prefix=args.normalize_prefix, use_mask=args.use_mask, mask_only=args.mask_only)
    val_ds   = ClipCocoDataset(args.val_pkl, args.prefix_length,
                               normalize_prefix=args.normalize_prefix, use_mask=args.use_mask, mask_only=args.mask_only)
    model = build_model(args)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_ds) // args.bs
    total_train_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_train_steps)

    best_val_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs}")
        train_one_epoch(train_ds, model, args, optimizer, scheduler)

        ckpt_path = os.path.join(args.out_dir, f"{args.prefix}-{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        print(">>> Computing validation loss")
        val_loss = validate_loss(val_ds, model)
        val_ppl = math.exp(min(50, val_loss))
        print(f"[Validation][epoch {epoch}] loss={val_loss:.6f}  ppl={val_ppl:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.out_dir, f"{args.prefix}-best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best; saved: {best_path}")

        metrics_path = os.path.join(args.out_dir, f"{args.prefix}_val_metrics_epoch_{epoch:03d}.json")
        with open(metrics_path, "w") as f:
            json.dump({"epoch": epoch, "val_loss": val_loss, "val_ppl": val_ppl}, f, indent=2)

    print(">>> Loading best checkpoint and generating validation captions & LeGrad maps")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f"{args.prefix}-best.pt"),
                                     map_location=next(model.parameters()).device))
    
    # [LEGRAD] Call the LeGrad function instead of the standard one
    csv_path = validate_and_save_csv_legrad(val_ds, model, args)
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()