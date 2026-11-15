"""
mp_wsol.py — Weakly supervised microplastic localisation with a Helmholtz-operator
feature regulariser, reading a Roboflow CSV export ("archive/train", "archive/valid",
optional "archive/test", each with _annotations.csv).

Version 2 (after the first real run): field-of-view filter OFF by default (it suits microscope
images, not dish photographs), hard negatives mined from glare/edges (dish rim), decision threshold
calibrated on validation, top-k pooling, BatchNorm + 2 extra convs, energy-normalised Helmholtz
penalty, pointing-game chance level, `summarize` command.

What changed relative to the original script
  * No synthetic flat "clear water" images. Classes are real 192x192 patches cut from
    the microscope images: POSITIVE = contains >=1 annotated particle,
    NEGATIVE = same images, no annotated particle within 8 px. Both must lie inside the
    illuminated field of view, so the dark corners cannot be used as a shortcut.
  * Training uses image-level (patch-level) labels only. Box annotations are used ONLY
    for evaluation (CorLoc, pointing game, precision/recall, AP@0.5, AP@[.5:.95]).
  * Roboflow copies of one photo ("x_jpg.rf.<hash>.jpg") are kept in the same split;
    train/valid overlap is removed from the test set. Split saved as a JSON manifest.
  * Validation set, fixed seeds, augmentation, early stopping, checkpoints, cosine LR.
  * Global-average-pooling head (~93 k parameters instead of 51 M).
  * Regulariser: none | helmholtz | l2 | tv | laplacian, on pre- or post-ReLU conv3.
  * CAM computed exactly from the GAP head (identical to Grad-CAM for this head),
    plus Eigen-CAM. Threshold tau tuned on validation images, reported on test images.
  * Variance-threshold sanity baseline, feature power-spectrum analysis, no random
    error bars. Everything is written to results.json / CSV / PNG in the run folder.

Commands (run `python mp_wsol.py <command> -h` for all options)
  audit     check the dataset and write audit.json + figures (no PyTorch needed)
  train     train, then evaluate on the test split
  evaluate  re-evaluate a trained run folder
  predict   boxes + overlays for a folder of new images
  export    ONNX export for deployment
  ablate    regulariser x seed sweep with a mean +/- s.d. summary table

Example (Windows paths with spaces need quotes):
  python mp_wsol.py audit --data-root "C:/Users/me/Documents/AIUB/AIUB/research/Arajit Shah/ML model/archive" --out runs/audit
  python mp_wsol.py train --data-root ".../archive" --out runs/helmholtz_s0 --reg helmholtz --reg-on pre
  python mp_wsol.py ablate --data-root ".../archive" --out runs/ablation --regs none,l2,tv,laplacian,helmholtz --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from functools import lru_cache

import cv2
import numpy as np

import mp_data as D

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:          # the `audit` command works without PyTorch
    torch = None
    nn = None
    Dataset = object


# =========================================================================== config
@dataclass
class Config:
    data_root: str = ""
    out: str = "runs/exp"
    seed: int = 0
    # data / patches
    patch: int = 128                 # crop size in original pixels
    input_size: int = 224            # network input size for a patch
    n_pos: int = 2                   # positive patches per image per epoch
    n_neg: int = 2                   # negative patches per image per epoch
    fov_min: float = 0.0             # >0 only for microscope images with a dark surround (off for dish photos)
    neg_margin: int = 8              # clearance (px) between a negative patch and any box
    dark_thr: int = 12               # grey level below which a pixel is outside the field
    min_gt_px: float = 3.0           # drop annotation boxes thinner than this
    hard_frac: float = 0.5           # share of negatives mined from glare/edge regions (dish rim)
    hard_cands: int = 16             # candidates examined per hard negative
    glare_thr: int = 150             # grey level counted as glare when mining hard negatives
    val_frac: float = 0.15           # used only when the archive has no separate folder
    test_frac: float = 0.15
    # model
    backbone: str = "optical"        # optical | resnet18
    pretrained: bool = False         # ImageNet weights for resnet18 (needs internet once)
    batchnorm: bool = True
    extra_convs: int = 2             # extra 3x3 convs (128->128) in stage 3; last conv is the regularised layer
    pool: str = "topk"               # avg | max | topk  (topk = mean of the top-k% positions per channel)
    topk_frac: float = 0.05
    reg: str = "helmholtz"           # none | helmholtz | l2 | tv | laplacian
    reg_on: str = "pre"              # pre | post (relative to the conv3 ReLU)
    lam: float = 0.1
    kappa: float = 1.5
    reg_norm: bool = True            # divide helmholtz/laplacian/tv by feature energy (no gain from shrinking)
    # optimisation
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    num_workers: int = 0             # keep 0 on Windows unless you know it works
    amp: bool = False
    device: str = "auto"
    # localisation
    taus: str = "0.2,0.3,0.4,0.5,0.6,0.7,0.8"
    min_box: int = 6
    fov_filter: bool = False         # ignore CAM outside the illuminated field (microscope images only)
    n_vis: int = 12


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y")


def add_config_args(ap: argparse.ArgumentParser, skip=()) -> None:
    for f in fields(Config):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, type=str2bool, default=f.default, metavar="BOOL")
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)


def config_from_args(ns: argparse.Namespace) -> Config:
    return Config(**{f.name: getattr(ns, f.name) for f in fields(Config) if hasattr(ns, f.name)})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(cfg: Config):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== image I/O
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)


@lru_cache(maxsize=512)
def load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(D.imread(path), cv2.COLOR_BGR2RGB)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def norm_stats(cfg=None):
    if cfg is not None and cfg.backbone == "resnet18" and cfg.pretrained:
        return IMAGENET_MEAN, IMAGENET_STD
    return MEAN, STD


def to_tensor(rgb: np.ndarray, cfg=None):
    mean, std = norm_stats(cfg)
    x = (rgb.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    if rng.random() < 0.5:
        rgb = rgb[::-1, :]
    rgb = np.rot90(rgb, int(rng.integers(0, 4)))
    x = rgb.astype(np.float32)
    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)          # contrast, brightness
    x = x * rng.uniform(0.92, 1.08, size=(1, 1, 3))                  # mild colour cast
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:
        x = x + rng.normal(0, 4, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


class PatchDataset(Dataset):
    def __init__(self, patches: list, cfg: Config, train: bool, seed: int = 0):
        self.patches, self.cfg, self.train, self.seed = patches, cfg, train, seed

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int):
        p = self.patches[i]
        rgb = D.crop(load_rgb(p.path), p)
        if self.train:
            rgb = augment(rgb, np.random.default_rng((self.seed, i)))
        rgb = cv2.resize(np.ascontiguousarray(rgb), (self.cfg.input_size, self.cfg.input_size),
                         interpolation=cv2.INTER_AREA if p.size > self.cfg.input_size else cv2.INTER_LINEAR)
        return to_tensor(rgb, self.cfg), p.label


def make_loader(patches, cfg: Config, train: bool, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(PatchDataset(patches, cfg, train, seed), batch_size=cfg.batch_size, shuffle=train,
                      num_workers=cfg.num_workers, generator=g, drop_last=False,
                      pin_memory=torch.cuda.is_available())


# =========================================================================== models
if torch is not None:
    def spatial_pool(u, mode: str, frac: float):
        """avg = GAP (CAM of Zhou et al.); max = GMP; topk = mean of the top-k% activations per
        channel, which keeps a small particle from being averaged away by the background."""
        if mode == "avg":
            return u.mean(dim=(2, 3))
        flat = u.flatten(2)
        if mode == "max":
            return flat.amax(dim=2)
        k = max(1, int(round(frac * flat.shape[2])))
        return flat.topk(k, dim=2).values.mean(dim=2)

    class OpticalCNN(nn.Module):
        """conv1 -> pool -> conv2 -> pool -> conv3a [-> conv3b ...] (last conv regularised; CAM layer)
        -> spatial pooling -> linear. Output stride 4."""

        def __init__(self, batchnorm: bool = True, extra_convs: int = 2, pool: str = "topk",
                     topk_frac: float = 0.05, num_classes: int = 2):
            super().__init__()

            def conv(cin, cout):
                layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not batchnorm)]
                if batchnorm:
                    layers.append(nn.BatchNorm2d(cout))
                return nn.Sequential(*layers)

            self.conv1, self.conv2 = conv(3, 32), conv(32, 64)
            chans = [64] + [128] * (1 + extra_convs)
            self.stage3 = nn.ModuleList([conv(chans[i], chans[i + 1]) for i in range(len(chans) - 1)])
            self.pool = nn.MaxPool2d(2)
            self.head = nn.Linear(128, num_classes)
            self.pool_mode, self.topk_frac, self.stride = pool, topk_frac, 4

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            for layer in self.stage3[:-1]:
                x = F.relu(layer(x))
            z = self.stage3[-1](x)                 # pre-activation of the regularised layer
            u = F.relu(z)                          # post-activation (CAM features)
            return self.head(spatial_pool(u, self.pool_mode, self.topk_frac)), z, u

    class ResNet18WSOL(nn.Module):
        """Baseline: ResNet-18 trunk, GAP head. z == u (block output is already post-ReLU)."""

        def __init__(self, pretrained: bool = False, pool: str = "topk", topk_frac: float = 0.05,
                     num_classes: int = 2):
            super().__init__()
            import torchvision
            self.pool_mode, self.topk_frac = pool, topk_frac
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            m = torchvision.models.resnet18(weights=weights)
            self.trunk = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
            self.head = nn.Linear(256, num_classes)   # layer3 (stride 16) keeps more spatial detail
            self.stride = 16

        def forward(self, x):
            u = self.trunk(x)
            return self.head(spatial_pool(u, self.pool_mode, self.topk_frac)), u, u


def build_model(cfg: Config):
    if cfg.backbone == "optical":
        return OpticalCNN(cfg.batchnorm, cfg.extra_convs, cfg.pool, cfg.topk_frac)
    if cfg.backbone == "resnet18":
        return ResNet18WSOL(cfg.pretrained, cfg.pool, cfg.topk_frac)
    raise ValueError(f"unknown backbone {cfg.backbone}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# =========================================================================== regularisers
def feature_regulariser(f, kind: str, kappa: float, normalise: bool = True):
    """Penalties on a (B,C,H,W) feature tensor. 'helmholtz' = mean((lap f + kappa^2 f)^2).
    With normalise=True the helmholtz/laplacian/tv penalties are divided by the feature energy
    (a Rayleigh quotient), so they constrain the *spectral shape* of the features and cannot be
    reduced by simply shrinking activations. Replicate padding avoids artificial border residuals."""
    if kind == "none":
        return f.new_zeros(())
    if kind == "l2":
        return (f ** 2).mean()
    eps = 1e-6
    if kind == "tv":
        tv = (f[..., 1:, :] - f[..., :-1, :]).abs().mean() + (f[..., :, 1:] - f[..., :, :-1]).abs().mean()
        return tv / (f.abs().mean() + eps) if normalise else tv
    c = f.shape[1]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=f.device, dtype=f.dtype)
    lap = F.conv2d(F.pad(f, (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3).expand(c, 1, 3, 3), groups=c)
    energy = (f ** 2).mean() + eps if normalise else 1.0
    if kind == "laplacian":
        return (lap ** 2).mean() / energy
    if kind == "helmholtz":
        return ((lap + (kappa ** 2) * f) ** 2).mean() / energy
    raise ValueError(f"unknown regulariser {kind}")


# =========================================================================== CAM
def cam_maps(model, x, method: str = "gradcam", target: int = 1):
    """Returns (B,h,w) non-normalised maps at feature resolution.
    gradcam: class activation map relu(sum_m W[target,m] * u_m) (Zhou et al., 2016). For a GAP
             head this equals Grad-CAM exactly; for max/top-k pooling it is the standard CAM.
    eigencam: projection of the activations on their first principal direction (sign
             chosen to correlate positively with Grad-CAM)."""
    with torch.no_grad():
        logits, _, u = model(x)
        w = model.head.weight[target]                              # (C,)
        grad = F.relu(torch.einsum("c,bchw->bhw", w, u))
        if method == "gradcam":
            return grad, logits
        b, c, h, wd = u.shape
        out = []
        for i in range(b):
            A = u[i].reshape(c, h * wd).T                          # (hw, C)
            _, _, vh = torch.linalg.svd(A, full_matrices=False)
            proj = (A @ vh[0]).reshape(h, wd)
            if torch.sum(proj * grad[i]) < 0:
                proj = -proj
            out.append(F.relu(proj))
        return torch.stack(out), logits


def full_image_cam(model, rgb: np.ndarray, cfg: Config, device, method: str) -> tuple[np.ndarray, float]:
    """CAM for a whole image at the same scale the patches were trained on. Returns a
    [0,1] map at the original resolution and the image-level P(microplastic)."""
    H, W = rgb.shape[:2]
    s = cfg.input_size / cfg.patch
    nh, nw = max(32, int(round(H * s))), max(32, int(round(W * s)))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = (-nh) % 16, (-nw) % 16
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = to_tensor(img, cfg).unsqueeze(0).to(device)
    cam, logits = cam_maps(model, x, method)
    cam = cam[0].float().cpu().numpy()
    cam = cv2.resize(cam, (nw + pw, nh + ph), interpolation=cv2.INTER_LINEAR)[:nh, :nw]
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    prob = float(torch.softmax(logits, 1)[0, 1])
    return D.normalise_cam(cam), prob


def patch_cam(model, patches, cfg: Config, device, method: str) -> list[np.ndarray]:
    out = []
    for i in range(0, len(patches), cfg.batch_size):
        chunk = patches[i:i + cfg.batch_size]
        xs = []
        for p in chunk:
            rgb = cv2.resize(np.ascontiguousarray(D.crop(load_rgb(p.path), p)), (cfg.input_size, cfg.input_size))
            xs.append(to_tensor(rgb, cfg))
        cams, _ = cam_maps(model, torch.stack(xs).to(device), method)
        for p, c in zip(chunk, cams.float().cpu().numpy()):
            out.append(D.normalise_cam(cv2.resize(c, (p.size, p.size), interpolation=cv2.INTER_LINEAR)))
    return out


# =========================================================================== train / eval loops
def classification_metrics(y, pred, prob) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
    y, pred = np.asarray(y), np.asarray(pred)
    return {"n": int(len(y)), "acc": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)) if len(set(y)) > 1 else 0.0,
            "auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else float("nan"),
            "confusion_matrix[[tn,fp],[fn,tp]]": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def run_epoch(model, loader, cfg: Config, device, optimizer=None, scaler=None) -> dict:
    train = optimizer is not None
    model.train(train)
    n, ce_sum, reg_sum = 0, 0.0, 0.0
    ys, preds, probs = [], [], []
    use_amp = cfg.amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=use_amp):
            logits, z, u = model(x)
            ce = F.cross_entropy(logits, y)
            reg = feature_regulariser((z if cfg.reg_on == "pre" else u).float(), cfg.reg, cfg.kappa, cfg.reg_norm)
            loss = ce + cfg.lam * reg
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        b = y.numel()
        n += b
        ce_sum += float(ce) * b
        reg_sum += float(reg) * b
        p = torch.softmax(logits.float(), 1)[:, 1]
        ys += y.tolist()
        preds += logits.argmax(1).tolist()
        probs += p.detach().cpu().tolist()
    if n == 0:
        return {"ce": float("nan"), "reg": float("nan")}
    out = {"ce": ce_sum / n, "reg": reg_sum / n, **classification_metrics(ys, preds, probs)}
    out["_y"], out["_p"] = ys, probs
    return out


def best_threshold(y, p) -> float:
    """Decision threshold on P(microplastic) that maximises MCC (chosen on validation)."""
    from sklearn.metrics import matthews_corrcoef
    y, p = np.asarray(y), np.asarray(p)
    grid = np.unique(np.round(p, 4))
    if len(grid) > 400:
        grid = np.quantile(p, np.linspace(0.01, 0.99, 199))
    scores = [(matthews_corrcoef(y, (p >= t).astype(int)) if len(set(y)) > 1 else 0.0, -abs(t - 0.5), t)
              for t in grid]
    return float(max(scores)[2]) if scores else 0.5


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def prepare_splits(cfg: Config) -> tuple[dict, dict]:
    os.makedirs(cfg.out, exist_ok=True)
    man_path = os.path.join(cfg.out, "split_manifest.json")
    if os.path.exists(man_path):
        return D.load_manifest(man_path), {"manifest": "reused"}
    data, logs = D.load_archive(cfg.data_root, cfg.min_gt_px)
    splits, info = D.make_splits(data, cfg.val_frac, cfg.test_frac, cfg.seed)
    D.save_manifest(splits, man_path)
    info["load_logs"] = logs
    return splits, info


def patch_kwargs(cfg: Config, **over) -> dict:
    kw = dict(patch=cfg.patch, n_pos=cfg.n_pos, n_neg=cfg.n_neg, fov_min=cfg.fov_min,
              neg_margin=cfg.neg_margin, dark_thr=cfg.dark_thr, hard_frac=cfg.hard_frac,
              hard_cands=cfg.hard_cands, glare_thr=cfg.glare_thr)
    kw.update(over)
    return kw


def patch_sets(splits: dict, cfg: Config, epoch_seed: int | None = None) -> dict:
    kw = patch_kwargs(cfg)
    out = {}
    if epoch_seed is not None:
        out["train"] = D.build_patch_set(splits["train"], epoch_seed, **kw)
    out["val"] = D.build_patch_set(splits["val"], 10_000 + cfg.seed, **kw)
    out["test"] = D.build_patch_set(splits["test"], 20_000 + cfg.seed, **kw)
    return out


def train(cfg: Config) -> str:
    set_seed(cfg.seed)
    device = get_device(cfg)
    splits, info = prepare_splits(cfg)
    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    with open(os.path.join(cfg.out, "split_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"[data] train {len(splits['train'])} img | val {len(splits['val'])} img | test {len(splits['test'])} img")
    if not splits["val"] or not splits["test"]:
        raise RuntimeError("Empty validation or test split - check split_info.json")

    fixed = patch_sets(splits, cfg)
    val_loader = make_loader(fixed["val"], cfg, False, cfg.seed)
    model = build_model(cfg).to(device)
    print(f"[model] {cfg.backbone}: {count_params(model):,} parameters | reg={cfg.reg}({cfg.reg_on}) "
          f"lam={cfg.lam} kappa={cfg.kappa} | device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = make_scaler(cfg.amp and device.type == "cuda")

    best_key, bad, hist = (-1.0, -1e9), 0, []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr_patches = D.build_patch_set(splits["train"], cfg.seed * 1000 + ep, **patch_kwargs(cfg))
        tr = run_epoch(model, make_loader(tr_patches, cfg, True, cfg.seed * 1000 + ep), cfg, device, opt, scaler)
        va = run_epoch(model, val_loader, cfg, device)
        sched.step()
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_ce": tr["ce"], "train_reg": tr["reg"],
               "train_acc": tr["acc"], "val_ce": va["ce"], "val_reg": va["reg"], "val_acc": va["acc"],
               "val_f1": va["f1"], "val_mcc": va["mcc"], "val_auc": va["auc"]}
        hist.append(row)
        print(f"ep {ep:03d} | train ce {tr['ce']:.4f} reg {tr['reg']:.4f} acc {tr['acc']:.3f} | "
              f"val ce {va['ce']:.4f} acc {va['acc']:.3f} f1 {va['f1']:.3f} mcc {va['mcc']:.3f} auc {va['auc']:.3f}")
        key = (np.nan_to_num(va["auc"]), -va["ce"])     # threshold-free; cut-off is calibrated later
        if key > best_key:
            best_key, bad = key, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": ep}, os.path.join(cfg.out, "best.pt"))
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[early stop] no val improvement for {cfg.patience} epochs")
                break
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": ep}, os.path.join(cfg.out, "last.pt"))
    with open(os.path.join(cfg.out, "train_time_s.txt"), "w") as f:
        f.write(f"{time.time() - t0:.1f}\n")
    with open(os.path.join(cfg.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    plot_history(hist, os.path.join(cfg.out, "training_curves.png"))
    print(f"[train] done in {time.time() - t0:.0f}s; best epoch (by val AUC) saved to best.pt")
    return cfg.out


# =========================================================================== evaluation
def load_run(run_dir: str, device):
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    cfg = Config(**ck["config"])
    cfg.out = run_dir
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("epoch")


def evaluate(run_dir: str, device_str: str = "auto", n_vis: int | None = None) -> dict:
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else
                          ("cpu" if device_str == "auto" else device_str))
    model, cfg, best_ep = load_run(run_dir, device)
    set_seed(cfg.seed)
    splits = D.load_manifest(os.path.join(run_dir, "split_manifest.json"))
    ps = patch_sets(splits, cfg)
    res = {"config": asdict(cfg), "best_epoch": best_ep, "parameters": count_params(model)}

    # ---- 1. classification on held-out patches ----------------------------------------
    va = run_epoch(model, make_loader(ps["val"], cfg, False, cfg.seed), cfg, device)
    te = run_epoch(model, make_loader(ps["test"], cfg, False, cfg.seed), cfg, device)
    thr = best_threshold(va["_y"], va["_p"])
    te_cal = classification_metrics(te["_y"], (np.asarray(te["_p"]) >= thr).astype(int), te["_p"])
    res["val_patch_classification"] = strip(va)
    res["test_patch_classification_argmax"] = strip(te)
    res["decision_threshold_from_val"] = thr
    res["test_patch_classification"] = te_cal
    te = te_cal
    tr_ref = D.build_patch_set(splits["train"], 30_000 + cfg.seed, **patch_kwargs(cfg, n_pos=1, n_neg=1))
    res["sanity_variance_baseline"] = D.variance_baseline(tr_ref, ps["test"])
    plot_confusion(te["confusion_matrix[[tn,fp],[fn,tp]]"], os.path.join(run_dir, "confusion_matrix.png"))

    # ---- 2. localisation: tune tau on val images, report on test images ----------------
    taus = [float(t) for t in cfg.taus.split(",")]
    res["localisation"] = {}
    for method in ("gradcam", "eigencam"):
        val_cams = [_image_cam(model, r, cfg, device, method) for r in splits["val"]]
        tune = {}
        for t in taus:
            preds = [D.cam_to_boxes(c, t, cfg.min_box, fov) for c, _, fov in val_cams]
            tune[t] = D.average_precision(preds, [r.boxes for r in splits["val"]], 0.5)
        tau = max(tune, key=lambda k: (np.nan_to_num(tune[k]), -k))
        test_cams = [_image_cam(model, r, cfg, device, method) for r in splits["test"]]
        preds = [D.cam_to_boxes(c, tau, cfg.min_box, fov) for c, _, fov in test_cams]
        peaks = [np.unravel_index(np.argmax(np.where(fov, c, -1)), c.shape)[::-1] for c, _, fov in test_cams]
        full = D.localisation_report(preds, [r.boxes for r in splits["test"]], peaks,
                                     [c.shape for c, _, _ in test_cams])
        # patch-level localisation (positive test patches)
        pos = [p for p in ps["test"] if p.label == 1]
        pc = patch_cam(model, pos, cfg, device, method)
        p_preds = [D.cam_to_boxes(c, tau, cfg.min_box) for c in pc]
        p_peaks = [np.unravel_index(np.argmax(c), c.shape)[::-1] for c in pc]
        patch_rep = D.localisation_report(p_preds, [p.boxes for p in pos], p_peaks, [c.shape for c in pc])
        res["localisation"][method] = {"tau_selected_on_val": tau, "val_ap50_by_tau": tune,
                                       "test_full_images": full, "test_patches": patch_rep}
        if method == "gradcam":
            save_overlays(splits["test"], test_cams, preds, os.path.join(run_dir, "overlays_test"),
                          n_vis if n_vis is not None else cfg.n_vis)

    # ---- 3. what does the regulariser do? feature power spectrum -----------------------
    res["feature_spectrum"] = feature_spectrum(model, ps["test"][:256], cfg, device,
                                               os.path.join(run_dir, "feature_spectrum.png"))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    _print_summary(res)
    return res


def _image_cam(model, rec, cfg: Config, device, method: str):
    rgb = load_rgb(rec.path)
    cam, prob = full_image_cam(model, rgb, cfg, device, method)
    fov = D.cached_fov(rec.path, cfg.dark_thr) if cfg.fov_filter else np.ones(cam.shape, bool)
    return cam, prob, fov


def _print_summary(res: dict) -> None:
    c = res["test_patch_classification"]
    print("\n=== TEST RESULTS ===")
    print(f"patch classification (threshold {res['decision_threshold_from_val']:.3f} from val): acc {c['acc']:.3f}  "
          f"f1 {c['f1']:.3f}  mcc {c['mcc']:.3f}  auc {c['auc']:.3f}  (n={c['n']})")
    v = res["sanity_variance_baseline"]
    print(f"variance-threshold baseline: acc {v['test_acc']:.3f}  auc {v['test_auc']:.3f}   "
          f"(if close to the model, the task is still too easy)")
    for m, r in res["localisation"].items():
        f = r["test_full_images"]
        if f.get("images_with_gt"):
            print(f"{m:8s} tau={r['tau_selected_on_val']:.2f} | full images: CorLoc {f['corloc']:.3f}  "
                  f"pointing {f['pointing_game']:.3f} (chance {f['pointing_game_chance']:.3f})  "
                  f"P {f['precision@0.5']:.3f}  R {f['recall@0.5']:.3f}  "
                  f"AP50 {f['ap50']:.3f}  AP50-95 {f['ap50_95']:.3f}")


def feature_spectrum(model, patches, cfg: Config, device, png: str) -> dict:
    """Radially averaged power spectrum of the regularised layer (pre and post ReLU).
    The Helmholtz penalty predicts suppressed DC/high frequencies and a preserved band
    near |omega| ~ kappa."""
    if not patches:
        return {}
    acc = {"z": None, "u": None}
    with torch.no_grad():
        for i in range(0, len(patches), cfg.batch_size):
            xs = [to_tensor(cv2.resize(np.ascontiguousarray(D.crop(load_rgb(p.path), p)),
                                       (cfg.input_size, cfg.input_size)), cfg) for p in patches[i:i + cfg.batch_size]]
            _, z, u = model(torch.stack(xs).to(device))
            for k, t in (("z", z), ("u", u)):
                pw = torch.fft.fftshift(torch.fft.fft2(t.float()).abs() ** 2, dim=(-2, -1)).mean(dim=(0, 1))
                acc[k] = pw if acc[k] is None else acc[k] + pw
    out = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for k, pw in acc.items():
        pw = pw.cpu().numpy()
        h, w = pw.shape
        yy, xx = np.indices(pw.shape)
        r = np.hypot((yy - h // 2) / h, (xx - w // 2) / w) * 2 * np.pi        # rad / pixel
        bins = np.linspace(0, np.pi, 33)
        idx = np.digitize(r.ravel(), bins)
        prof = np.array([pw.ravel()[idx == j].mean() if np.any(idx == j) else np.nan for j in range(1, len(bins))])
        centres = 0.5 * (bins[1:] + bins[:-1])
        out[k] = {"omega_rad_per_px": centres.round(4).tolist(), "power": prof.tolist()}
        plt.semilogy(centres, prof, label=f"{'pre' if k == 'z' else 'post'}-ReLU")
    # zero ring of the discrete operator along an axis: 4 sin^2(w/2) = kappa^2
    if cfg.kappa < 2:
        w0 = 2 * np.arcsin(cfg.kappa / 2)
        plt.axvline(w0, ls="--", c="grey", label=f"operator zero, kappa={cfg.kappa}")
    plt.xlabel("spatial frequency |omega| (rad / feature pixel)")
    plt.ylabel("mean power")
    plt.title(f"Feature spectrum ({cfg.reg}, {cfg.reg_on})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()
    return out


# =========================================================================== figures
def plot_history(hist: list[dict], png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_ce"] for h in hist], label="train CE")
    ax[0].plot(ep, [h["val_ce"] for h in hist], label="val CE")
    ax[0].set_title("Cross-entropy")
    ax[1].plot(ep, [h["train_reg"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_reg"] for h in hist], label="val")
    ax[1].set_yscale("log")
    ax[1].set_title("Regulariser")
    ax[2].plot(ep, [h["train_acc"] for h in hist], label="train acc")
    ax[2].plot(ep, [h["val_acc"] for h in hist], label="val acc")
    ax[2].plot(ep, [h["val_mcc"] for h in hist], label="val MCC")
    ax[2].set_title("Validation metrics")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()


def plot_confusion(cm, png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=14,
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["background", "microplastic"])
    plt.yticks([0, 1], ["background", "microplastic"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test patches")
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()


def draw_overlay(rgb, cam, preds, gts=None, fov=None) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    out = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)
    if fov is not None:
        cnts, _ = cv2.findContours(fov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)
    if gts is not None:
        for b in np.asarray(gts).astype(int):
            cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    for b in preds:
        x0, y0, x1, y1 = map(int, b[:4])
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, f"{b[4]:.2f}", (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(out, "green = annotation, red = prediction" if gts is not None else "red = prediction",
                (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def save_overlays(recs, cams, preds, folder: str, n: int) -> None:
    os.makedirs(folder, exist_ok=True)
    for r, (cam, _, fov), p in list(zip(recs, cams, preds))[:n]:
        out = draw_overlay(load_rgb(r.path), cam, p, r.boxes, fov)
        cv2.imencode(".jpg", out)[1].tofile(os.path.join(folder, os.path.splitext(r.name)[0] + "_overlay.jpg"))


# =========================================================================== predict / export
def predict(run_dir: str, images: str, out: str, tau: float | None, method: str, device_str: str) -> None:
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else
                          ("cpu" if device_str == "auto" else device_str))
    model, cfg, _ = load_run(run_dir, device)
    if tau is None:
        try:
            with open(os.path.join(run_dir, "results.json")) as f:
                tau = json.load(f)["localisation"][method]["tau_selected_on_val"]
        except (OSError, KeyError):
            tau = 0.5
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(images) if f.lower().endswith(D.IMG_EXT))
    results = []
    for fn in files:
        path = os.path.join(images, fn)
        rgb = load_rgb(path)
        cam, prob = full_image_cam(model, rgb, cfg, device, method)
        fov = D.field_of_view(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cfg.dark_thr) if cfg.fov_filter else None
        boxes = D.cam_to_boxes(cam, tau, cfg.min_box, fov)
        results.append({"file": fn, "p_microplastic_image": prob, "tau": tau,
                        "boxes_xyxy_score": boxes.round(2).tolist(), "count": int(len(boxes))})
        ov = draw_overlay(rgb, cam, boxes, None, fov)
        cv2.imencode(".jpg", ov)[1].tofile(os.path.join(out, os.path.splitext(fn)[0] + "_pred.jpg"))
    with open(os.path.join(out, "predictions.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"[predict] {len(files)} images -> {out}")


def export_onnx(run_dir: str) -> None:
    model, cfg, _ = load_run(run_dir, torch.device("cpu"))
    x = torch.randn(1, 3, cfg.input_size, cfg.input_size)
    path = os.path.join(run_dir, "model.onnx")
    torch.onnx.export(model, x, path, input_names=["image"], output_names=["logits", "z", "u"],
                      dynamic_axes={"image": {0: "b", 2: "h", 3: "w"}, "u": {0: "b", 2: "fh", 3: "fw"},
                                    "z": {0: "b", 2: "fh", 3: "fw"}, "logits": {0: "b"}}, opset_version=17)
    with open(os.path.join(run_dir, "cam_head.json"), "w") as f:
        json.dump({"weight_class1": model.head.weight[1].tolist(), "input_mean": norm_stats(cfg)[0].tolist(),
                   "input_std": norm_stats(cfg)[1].tolist(), "patch": cfg.patch, "input_size": cfg.input_size}, f)
    print(f"[export] {path}  (CAM = relu(sum_m w_m * u_m) using cam_head.json)")


# =========================================================================== ablation
ABL_KEYS = [("test_patch_classification", "f1"), ("test_patch_classification", "mcc"),
            ("test_patch_classification", "auc")]
LOC_KEYS = ["corloc", "pointing_game", "pointing_game_chance", "precision@0.5", "recall@0.5", "ap50", "ap50_95"]
CFG_KEYS = ["backbone", "pretrained", "reg", "reg_on", "reg_norm", "lam", "kappa", "pool", "patch",
            "hard_frac", "extra_convs", "seed"]


def summarize(run_dirs: list[str], out_csv: str) -> None:
    """Collect results.json from any set of run folders into one CSV (one row per run) plus a
    grouped mean +/- s.d. table over seeds. Send these two CSVs back for the manuscript."""
    import glob as _glob
    dirs = sorted({d for pat in run_dirs for d in _glob.glob(pat)
                   if os.path.exists(os.path.join(d, "results.json"))})
    if not dirs:
        raise SystemExit("no run folders with results.json matched")
    rows = []
    for d in dirs:
        with open(os.path.join(d, "results.json")) as f:
            r = json.load(f)
        c = r["config"]
        row = {"run": d, **{k: c.get(k) for k in CFG_KEYS}, "params": r.get("parameters"),
               "best_epoch": r.get("best_epoch"), "threshold": r.get("decision_threshold_from_val")}
        for k in ("acc", "precision", "recall", "f1", "mcc", "auc"):
            row[k] = r["test_patch_classification"][k]
        row["varbase_auc"] = r["sanity_variance_baseline"]["test_auc"]
        for m in ("gradcam", "eigencam"):
            row[f"{m}_tau"] = r["localisation"][m]["tau_selected_on_val"]
            for k in LOC_KEYS:
                row[f"{m}_{k}"] = r["localisation"][m]["test_full_images"].get(k, float("nan"))
        tt = os.path.join(d, "train_time_s.txt")
        row["train_time_s"] = float(open(tt).read()) if os.path.exists(tt) else float("nan")
        rows.append(row)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    group_keys = [k for k in CFG_KEYS if k != "seed"]
    metrics = [k for k in rows[0] if k not in ("run", "params", "seed", *group_keys)]
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    grouped = os.path.splitext(out_csv)[0] + "_grouped.csv"
    with open(grouped, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(group_keys + ["n_seeds"] + metrics)
        for key, sub in groups.items():
            vals = []
            for m in metrics:
                x = np.array([np.nan if s_[m] is None else float(s_[m]) for s_ in sub], float)
                ok = x[~np.isnan(x)]
                if len(ok) == 0:
                    vals.append("nan")
                    continue
                sd = float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0
                vals.append(f"{float(np.mean(ok)):.4f} ± {sd:.4f}")
            w.writerow(list(key) + [len(sub)] + vals)
    print(f"[summarize] {len(rows)} runs -> {out_csv} and {grouped}")


def ablate(base: Config, regs: list[str], seeds: list[int]) -> None:
    rows = []
    for reg in regs:
        for seed in seeds:
            cfg = Config(**{**asdict(base), "reg": reg, "seed": seed,
                            "out": os.path.join(base.out, f"{reg}_s{seed}")})
            if not os.path.exists(os.path.join(cfg.out, "results.json")):
                train(cfg)
                evaluate(cfg.out, cfg.device)
            with open(os.path.join(cfg.out, "results.json")) as f:
                r = json.load(f)
            row = {"reg": reg, "seed": seed}
            for a, b in ABL_KEYS:
                row[b] = r[a][b]
            for m in ("gradcam", "eigencam"):
                for k in LOC_KEYS:
                    row[f"{m}_{k}"] = r["localisation"][m]["test_full_images"].get(k, float("nan"))
            rows.append(row)
    with open(os.path.join(base.out, "ablation_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    metrics = [k for k in rows[0] if k not in ("reg", "seed")]
    with open(os.path.join(base.out, "ablation_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reg", "n_seeds"] + metrics)
        for reg in regs:
            sub = [r for r in rows if r["reg"] == reg]
            w.writerow([reg, len(sub)] + [f"{np.nanmean([r[m] for r in sub]):.4f} ± {np.nanstd([r[m] for r in sub], ddof=1) if len(sub) > 1 else 0:.4f}"
                                          for m in metrics])
    print(f"[ablate] summary -> {os.path.join(base.out, 'ablation_summary.csv')}")


# =========================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="dataset checks (no PyTorch needed)")
    a.add_argument("--data-root", required=True)
    a.add_argument("--out", default="runs/audit")
    a.add_argument("--dark-thr", type=int, default=12)

    t = sub.add_parser("train", help="train + evaluate")
    add_config_args(t)

    e = sub.add_parser("evaluate", help="evaluate an existing run folder")
    e.add_argument("--run", required=True)
    e.add_argument("--device", default="auto")
    e.add_argument("--n-vis", type=int, default=None)

    p = sub.add_parser("predict", help="predict boxes for a folder of images")
    p.add_argument("--run", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="predictions")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--method", default="gradcam", choices=["gradcam", "eigencam"])
    p.add_argument("--device", default="auto")

    x = sub.add_parser("export", help="ONNX export")
    x.add_argument("--run", required=True)

    sm = sub.add_parser("summarize", help="collect results.json of many runs into CSV tables")
    sm.add_argument("--runs", nargs="+", required=True, help='folders or patterns, e.g. "runs\\*" "runs\\ablation\\*"')
    sm.add_argument("--out", default="runs/summary.csv")

    b = sub.add_parser("ablate", help="regulariser x seed sweep")
    add_config_args(b, skip=("reg", "seed"))
    b.add_argument("--regs", default="none,l2,tv,laplacian,helmholtz")
    b.add_argument("--seeds", default="0,1,2,3,4")

    ns = ap.parse_args()
    if ns.cmd == "audit":
        rep = D.audit_archive(ns.data_root, ns.out, ns.dark_thr)
        print(json.dumps({"per_split": rep["per_split"], "warnings": rep["warnings"]}, indent=1))
        print(f"[audit] full report and figures in {ns.out}")
        return
    if ns.cmd == "summarize":
        summarize(ns.runs, ns.out)
        return
    if torch is None:
        raise SystemExit("PyTorch is required for this command: pip install torch torchvision")
    if ns.cmd == "train":
        cfg = config_from_args(ns)
        if not cfg.data_root:
            raise SystemExit("--data-root is required")
        train(cfg)
        evaluate(cfg.out, cfg.device)
    elif ns.cmd == "evaluate":
        evaluate(ns.run, ns.device, ns.n_vis)
    elif ns.cmd == "predict":
        predict(ns.run, ns.images, ns.out, ns.tau, ns.method, ns.device)
    elif ns.cmd == "export":
        export_onnx(ns.run)
    elif ns.cmd == "summarize":
        summarize(ns.runs, ns.out)
    elif ns.cmd == "ablate":
        base = config_from_args(ns)
        if not base.data_root:
            raise SystemExit("--data-root is required")
        ablate(base, [r.strip() for r in ns.regs.split(",") if r.strip()],
               [int(s) for s in ns.seeds.split(",") if s.strip()])


if __name__ == "__main__":
    main()
