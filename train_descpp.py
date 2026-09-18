"""
Train DescPP on MegaDepth.

Usage:
    python train.py --feature orb
    python train.py --feature superpoint
    python train.py --feature alike --batch-size 8
    python train.py --feature sift --resume runs/DescPP_sift/last.pt
"""
import argparse
import copy
import csv
import math
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.dataset import MegaDepth, custom_collate
from lib.features import EXTRACTORS, build_extractor
from lib.loss import LossProcessor, boost_objective
from model.DescPP import build_model


# ============================================================ config / args
def parse_args():
    ap = argparse.ArgumentParser(description="Train DescPP")
    ap.add_argument("--feature", required=True, choices=sorted(EXTRACTORS))
    ap.add_argument("--config", default="train_config.yml")
    ap.add_argument("--checkpoint-dir", default=None, help="Defaults to <checkpoint_root>/DescPP_<feature>")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-val", action="store_true")
    ap.add_argument("--resume", default=None, help="Path to a last.pt checkpoint to resume from")
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.feature not in cfg["features"]:
        raise KeyError(f"Feature '{args.feature}' not found in {args.config}")

    tr = cfg["train"]
    for key, val in [("max_epoch", args.epochs), ("batch_size", args.batch_size),
                     ("init_lr", args.lr), ("num_workers", args.num_workers), ("seed", args.seed)]:
        if val is not None:
            tr[key] = val
    if args.run_val:
        tr["run_val"] = True

    cfg["feature_name"] = args.feature
    cfg["feature"] = cfg["features"][args.feature]
    cfg["checkpoint_dir"] = args.checkpoint_dir or os.path.join(
        cfg["paths"]["checkpoint_root"], f"DescPP_{args.feature}")
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================ LR schedule
def lr_at(step, total_steps, cfg_train):
    """Linear warmup followed by cosine decay to init_lr * min_lr_ratio."""
    base, warmup = cfg_train["init_lr"], cfg_train["warmup_steps"]
    if step < warmup:
        return base * (step + 1) / warmup
    lr_end = base * cfg_train["min_lr_ratio"]
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return lr_end + 0.5 * (base - lr_end) * (1 + math.cos(math.pi * progress))


# ============================================================ data
def make_dataset(cfg, extractor, train):
    p, d, feat = cfg["paths"], cfg["data"], cfg["feature"]
    return MegaDepth(
        scene_list_path=p["train_scene_path"] if train else p["valid_scene_path"],
        scene_info_path=p["scene_info_path"],
        base_path=p["base_path"],
        extractor=extractor,
        binary_desc=feat["descriptor"] == "binary",
        motion_blur=feat.get("motion_blur") if train else None,
        min_overlap_ratio=d["overlap_min"],
        max_overlap_ratio=d["overlap_max"],
        pairs_per_scene=d["max_pairs_per_scene"] if train else d["val_pairs_per_scene"],
        kps_per_image=d["kps_per_image"],
        crop_image_size=d["crop_image_size"],
    )


def make_loader(dataset, cfg, train):
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=train,
        collate_fn=custom_collate,
    )


# ============================================================ epoch loop
def run_batch(batch, model, loss_proc, device):
    """Returns the concatenated (origin_ap_loss, boost_ap_loss) over all valid pairs in the batch."""
    origin, boost = [], []
    for b in range(len(batch["kp0"])):
        sample = {k: v[b].to(device, non_blocking=True) for k, v in batch.items()}
        o, bst = loss_proc.forward_pair(model, sample)
        if o is not None:
            origin.append(o)
            boost.append(bst)
    if not origin:
        return None, None
    return torch.cat(origin), torch.cat(boost)


def train_one_epoch(loader, model, optimizer, loss_proc, cfg, epoch, global_step, total_steps,
                    device, step_log):
    model.train()
    w = cfg["feature"]["boost_weight"]
    hist = {"loss": [], "match": [], "boost": []}
    pbar = tqdm(loader, desc=f"Train {epoch + 1}", ncols=110)

    for step_in_epoch, batch in enumerate(pbar):
        origin, boost = run_batch(batch, model, loss_proc, device)
        if origin is None:
            continue

        loss, match_loss, boost_loss = boost_objective(origin, boost, boost_weight=w)

        lr = lr_at(global_step, total_steps, cfg["train"])
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        global_step += 1

        vals = (loss.item(), match_loss.item(), boost_loss.item())
        for k, v in zip(hist, vals):
            hist[k].append(v)
        step_log.writerow([epoch + 1, global_step, step_in_epoch, *vals, lr])
        pbar.set_postfix(loss=f"{vals[0]:.4f}", ap=f"{1 - vals[1]:.3f}", boost=f"{vals[2]:.3f}",
                         lr=f"{lr:.1e}")

    means = {k: float(np.mean(v)) if v else float("nan") for k, v in hist.items()}
    return means, global_step


@torch.no_grad()
def validate(loader, model, loss_proc, device):
    """Returns the mean AP of the enhanced descriptors."""
    model.eval()
    aps = []
    for batch in tqdm(loader, desc="Val", ncols=110):
        _, boost = run_batch(batch, model, loss_proc, device)
        if boost is not None:
            aps.append(1 - boost.mean().item())
    return float(np.mean(aps)) if aps else float("nan")


# ============================================================ checkpoint
def save_checkpoint(path, model, optimizer, epoch, global_step, total_steps, cfg):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "total_steps": total_steps,
        "feature": cfg["feature_name"],
        "model_cfg": cfg["feature"]["model"],
        "config": cfg,
    }, path)


def fmt(x):
    return f"{x:.6f}" if x is not None and np.isfinite(x) else ""


# ============================================================ main
def main():
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    print(yaml.safe_dump({k: v for k, v in cfg.items() if k != "features"}, sort_keys=False,
                         allow_unicode=True))
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False, allow_unicode=True)

    # ---- model / loss / optimizer
    model = build_model(cfg["feature"]["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["init_lr"])
    loss_proc = LossProcessor(descriptor_type=cfg["feature"]["descriptor"], **cfg["loss"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DescPP ({cfg['feature_name']}): {n_params:,} parameters, "
          f"{n_params * 4 / 1024 ** 2:.2f} MB (fp32)")

    start_epoch, global_step, total_steps = 0, 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if ck["feature"] != cfg["feature_name"]:
            raise ValueError(f"Checkpoint was trained with feature '{ck['feature']}'.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, global_step, total_steps = ck["epoch"] + 1, ck["global_step"], ck["total_steps"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}.")

    # ---- logs (appended when resuming)
    mode = "a" if args.resume else "w"
    step_f = open(os.path.join(ckpt_dir, "step_log.csv"), mode, newline="")
    epoch_f = open(os.path.join(ckpt_dir, "train_log.csv"), mode, newline="")
    step_log, epoch_log = csv.writer(step_f), csv.writer(epoch_f)
    if not args.resume:
        step_log.writerow(["epoch", "global_step", "step_in_epoch", "loss", "match_loss", "boost_loss", "lr"])
        epoch_log.writerow(["epoch", "train_loss", "train_match_loss", "train_boost_loss", "val_ap"])

    # ---- data
    extractor = build_extractor(cfg["feature_name"], **cfg["feature"].get("extractor", {}))
    train_set = make_dataset(cfg, extractor, train=True)
    val_loader = None
    if cfg["train"]["run_val"]:
        val_set = make_dataset(cfg, extractor, train=False)
        val_set.build_dataset(desc="Building validation set")
        val_loader = make_loader(val_set, cfg, train=False)

    max_epoch = cfg["train"]["max_epoch"]
    for epoch in range(start_epoch, max_epoch):
        print(f"\n[Epoch {epoch + 1}/{max_epoch}]")
        if epoch == start_epoch or cfg["data"]["rebuild_every_epoch"]:
            train_set.build_dataset(desc="Building training set")
        if total_steps is None:
            total_steps = max_epoch * (len(train_set) // cfg["train"]["batch_size"])
        train_loader = make_loader(train_set, cfg, train=True)

        tr, global_step = train_one_epoch(train_loader, model, optimizer, loss_proc, cfg, epoch,
                                          global_step, total_steps, device, step_log)
        val_ap = validate(val_loader, model, loss_proc, device) if val_loader else None

        # Weights only, for evaluation.
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:03d}.pt"))
        # Full training state, for --resume.
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epoch,
                        global_step, total_steps, cfg)

        epoch_log.writerow([epoch + 1, fmt(tr["loss"]), fmt(tr["match"]), fmt(tr["boost"]), fmt(val_ap)])
        step_f.flush()
        epoch_f.flush()
        print(f"  train loss {tr['loss']:.4f} | AP {1 - tr['match']:.4f} | boost {tr['boost']:.4f}"
              + (f" | val AP {val_ap:.4f}" if val_ap is not None else ""))

    step_f.close()
    epoch_f.close()


if __name__ == "__main__":
    main()