"""
mp_data.py — data handling, auditing, patch sampling and metrics for the
microplastic weakly-supervised localisation pipeline.

Expected layout (Roboflow "TensorFlow Object Detection CSV" export):

    archive/
      train/   _annotations.csv  + *.jpg
      valid/   _annotations.csv  + *.jpg
      test/    _annotations.csv  + *.jpg      (optional)

_annotations.csv columns: filename,width,height,class,xmin,ymin,xmax,ymax

This module needs only numpy, pandas, opencv-python, scikit-learn and
matplotlib (no PyTorch), so the audit can run on any machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CSV_NAME = "_annotations.csv"
REQUIRED_COLS = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
SPLIT_NAMES = ("train", "valid", "val", "test")
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+(\.[A-Za-z]+)$")


# =========================================================================== records
@dataclass
class ImageRecord:
    path: str
    split: str                      # folder it came from: train / valid / test
    source: str                     # original image id (Roboflow copies share it)
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy, pixels
    classes: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def source_id(filename: str) -> str:
    """'c--188-_jpg.rf.3bd1...9db.jpg' -> 'c--188-_jpg'. Roboflow augmentations of the
    same photo share this id, so they must never be split across train/val/test."""
    base = os.path.basename(filename)
    stripped = _RF_SUFFIX.sub("", base)
    return stripped if stripped != base else os.path.splitext(base)[0]


def imread(path: str) -> np.ndarray:
    """Unicode-safe image read (works with spaces / non-ASCII Windows paths). Returns BGR uint8."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def find_split_dirs(root: str) -> dict[str, str]:
    out = {}
    for name in SPLIT_NAMES:
        d = os.path.join(root, name)
        if os.path.isdir(d):
            out["valid" if name == "val" else name] = d
    if not out:
        raise FileNotFoundError(
            f"No train/valid/test folders found in '{root}'. Point --data-root at the 'archive' folder.")
    return out


def load_split(split_dir: str, split: str, min_box_px: float = 3.0, log: dict | None = None) -> list[ImageRecord]:
    """Read one split folder. Boxes are clipped to the image; boxes thinner than
    `min_box_px` on either side are dropped (counted in `log`)."""
    log = log if log is not None else {}
    csv_path = os.path.join(split_dir, CSV_NAME)
    on_disk = {f for f in os.listdir(split_dir) if f.lower().endswith(IMG_EXT)}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {CSV_NAME} in {split_dir}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} lacks columns {missing_cols}; found {list(df.columns)}")
    df["filename"] = df["filename"].astype(str).str.strip()

    in_csv = set(df["filename"])
    log["csv_rows"] = int(len(df))
    log["images_on_disk"] = len(on_disk)
    log["csv_files_missing_on_disk"] = sorted(in_csv - on_disk)[:50]
    log["n_csv_files_missing_on_disk"] = len(in_csv - on_disk)
    log["unannotated_images_on_disk"] = len(on_disk - in_csv)
    log["unannotated_examples"] = sorted(on_disk - in_csv)[:20]

    dropped_small, clipped = 0, 0
    records = []
    for fname, g in df[df["filename"].isin(on_disk)].groupby("filename", sort=True):
        w, h = int(g["width"].iloc[0]), int(g["height"].iloc[0])
        b = g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)
        b_clip = b.copy()
        b_clip[:, [0, 2]] = np.clip(b_clip[:, [0, 2]], 0, w)
        b_clip[:, [1, 3]] = np.clip(b_clip[:, [1, 3]], 0, h)
        clipped += int((np.abs(b_clip - b) > 0.5).any(axis=1).sum())
        keep = ((b_clip[:, 2] - b_clip[:, 0]) >= min_box_px) & ((b_clip[:, 3] - b_clip[:, 1]) >= min_box_px)
        dropped_small += int((~keep).sum())
        records.append(ImageRecord(path=os.path.join(split_dir, fname), split=split, source=source_id(fname),
                                   width=w, height=h, boxes=b_clip[keep],
                                   classes=g["class"].astype(str).to_numpy()[keep].tolist()))
    log["annotated_images"] = len(records)
    log["boxes_kept"] = int(sum(len(r.boxes) for r in records))
    log["boxes_dropped_too_small"] = dropped_small
    log["boxes_clipped_to_image"] = clipped
    log["class_names"] = sorted({c for r in records for c in r.classes})
    return records


def load_archive(root: str, min_box_px: float = 3.0) -> tuple[dict[str, list[ImageRecord]], dict]:
    logs = {}
    data = {}
    for split, d in find_split_dirs(root).items():
        logs[split] = {}
        data[split] = load_split(d, split, min_box_px, logs[split])
    return data, logs


# =========================================================================== splitting
def group_split(records: list[ImageRecord], frac: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split records into (rest, held_out) so that all copies of one source go to the same side."""
    groups = defaultdict(list)
    for r in records:
        groups[r.source].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_hold = max(1, int(round(len(keys) * frac))) if len(keys) > 1 else 0
    hold = set(keys[:n_hold])
    rest = [r for k in keys if k not in hold for r in groups[k]]
    held = [r for k in keys if k in hold for r in groups[k]]
    return rest, held


def make_splits(data: dict[str, list[ImageRecord]], val_frac: float, test_frac: float, seed: int,
                drop_overlap: bool = True) -> tuple[dict[str, list[ImageRecord]], dict]:
    """
    train/val/test policy:
      * 'valid' and 'test' folders are held out from training.
      * If a 'test' folder exists:   test = test folder, val = valid folder (or a group split of train).
      * If only 'valid' exists:      test = valid folder, val = group split of train (val_frac).
      * If only 'train' exists:      val and test are both group splits of train.
      * Sources that also occur in train are removed from val/test (drop_overlap) to stop leakage.
    """
    info = {}
    train = list(data.get("train", []))
    if "test" in data:
        test = list(data["test"])
        if "valid" in data:
            val = list(data["valid"])
        else:
            train, val = group_split(train, val_frac, seed)
    elif "valid" in data:
        test = list(data["valid"])
        train, val = group_split(train, val_frac, seed)
    else:
        train, test = group_split(train, test_frac, seed)
        train, val = group_split(train, val_frac / (1 - test_frac), seed + 1)

    train_src = {r.source for r in train}
    for name, recs in (("val", val), ("test", test)):
        overlap = sorted({r.source for r in recs} & train_src)
        info[f"{name}_sources_overlapping_train"] = len(overlap)
        info[f"{name}_overlap_examples"] = overlap[:20]
        if drop_overlap and overlap:
            recs[:] = [r for r in recs if r.source not in train_src]
    vt_overlap = {r.source for r in val} & {r.source for r in test}
    if drop_overlap and vt_overlap:
        val = [r for r in val if r.source not in vt_overlap]
    info["val_test_overlap_dropped"] = len(vt_overlap)
    out = {"train": train, "val": val, "test": test}
    for k, v in out.items():
        info[f"{k}_images"] = len(v)
        info[f"{k}_sources"] = len({r.source for r in v})
        info[f"{k}_boxes"] = int(sum(len(r.boxes) for r in v))
    return out, info


def save_manifest(splits: dict[str, list[ImageRecord]], path: str) -> None:
    js = {k: [{"path": r.path, "split_folder": r.split, "source": r.source, "width": r.width,
               "height": r.height, "boxes": r.boxes.round(1).tolist(), "classes": r.classes} for r in v]
          for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=1)


def load_manifest(path: str) -> dict[str, list[ImageRecord]]:
    with open(path, encoding="utf-8") as f:
        js = json.load(f)
    return {k: [ImageRecord(path=d["path"], split=d["split_folder"], source=d["source"], width=d["width"],
                            height=d["height"], boxes=np.asarray(d["boxes"], np.float32).reshape(-1, 4),
                            classes=d["classes"]) for d in v] for k, v in js.items()}


# =========================================================================== field of view
def field_of_view(img_bgr: np.ndarray, dark_thr: int = 12, min_dark_frac: float = 0.05,
                  rim_px: int = 12) -> np.ndarray:
    """Boolean mask of the illuminated microscope field. Pixels outside the circular
    field are near-black; if <5% of the image is dark the whole frame is returned.
    The mask is eroded by `rim_px` so the bright ring/rim (a glare source) is excluded."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 5)
    dark = g < dark_thr
    if dark.mean() < min_dark_frac:
        return np.ones(g.shape, bool)
    lit = (~dark).astype(np.uint8)
    lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    if n <= 1:
        return np.ones(g.shape, bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)       # fill holes (particles, ring)
    if rim_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_px + 1, 2 * rim_px + 1))
        filled = cv2.erode(filled, k)
    return filled.astype(bool)


@lru_cache(maxsize=4096)
def cached_fov(path: str, dark_thr: int = 12) -> np.ndarray:
    return field_of_view(imread(path), dark_thr)


@lru_cache(maxsize=512)
def cached_hardness(path: str, glare_thr: int = 150) -> np.ndarray:
    """Per-pixel 'distractor' map used to mine hard negatives: bright glare
    (container rims, reflections) and strong edges (dish walls, scratches)."""
    g = cv2.cvtColor(imread(path), cv2.COLOR_BGR2GRAY)
    glare = (g > glare_thr).astype(np.float32)
    edges = (cv2.Canny(g, 60, 160) > 0).astype(np.float32)
    return glare + 0.5 * edges


# =========================================================================== patches
@dataclass
class Patch:
    path: str
    x0: int
    y0: int
    size: int
    label: int                                  # 1 = contains microplastic, 0 = background
    boxes: np.ndarray                           # GT boxes in patch coordinates (xyxy)
    source: str = ""


def _overlaps_any(x0, y0, s, boxes, margin) -> bool:
    if len(boxes) == 0:
        return False
    b = boxes
    return bool(np.any((b[:, 0] - margin < x0 + s) & (b[:, 2] + margin > x0) &
                       (b[:, 1] - margin < y0 + s) & (b[:, 3] + margin > y0)))


def boxes_in_patch(boxes: np.ndarray, x0: int, y0: int, s: int, min_keep: float = 0.5) -> np.ndarray:
    """GT boxes clipped to the patch, kept if >= min_keep of their area lies inside."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    c = boxes.copy()
    c[:, [0, 2]] = np.clip(c[:, [0, 2]], x0, x0 + s)
    c[:, [1, 3]] = np.clip(c[:, [1, 3]], y0, y0 + s)
    a0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    a1 = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    keep = a1 >= min_keep * np.maximum(a0, 1e-6)
    out = c[keep] - np.array([x0, y0, x0, y0], np.float32)
    return out.astype(np.float32)


def sample_patches(rec: ImageRecord, rng: np.random.Generator, patch: int, n_pos: int, n_neg: int,
                   fov: np.ndarray | None, fov_min: float = 0.6, neg_margin: int = 8,
                   pos_margin: int = 4, tries: int = 200, hard: np.ndarray | None = None,
                   hard_frac: float = 0.0, hard_cands: int = 16) -> list[Patch]:
    """Positive patches fully contain >=1 annotated particle; negative patches touch no
    annotated particle (with `neg_margin` px clearance). Both must lie >= fov_min inside the
    illuminated field so that 'black area outside the microscope' cannot be used as a cue."""
    W, H = rec.width, rec.height
    s = int(min(patch, W, H))
    out: list[Patch] = []

    def fov_ok(x0, y0):
        return fov is None or fov[y0:y0 + s, x0:x0 + s].mean() >= fov_min

    # positives
    if len(rec.boxes):
        order = rng.permutation(len(rec.boxes))
        made = 0
        for t in range(tries):
            if made >= n_pos:
                break
            bx = rec.boxes[order[t % len(order)]]
            lo_x, hi_x = int(np.ceil(bx[2] + pos_margin - s)), int(np.floor(bx[0] - pos_margin))
            lo_y, hi_y = int(np.ceil(bx[3] + pos_margin - s)), int(np.floor(bx[1] - pos_margin))
            lo_x, hi_x = max(lo_x, 0), min(hi_x, W - s)
            lo_y, hi_y = max(lo_y, 0), min(hi_y, H - s)
            if lo_x > hi_x or lo_y > hi_y:
                continue                                    # box larger than patch
            x0, y0 = int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1))
            if not fov_ok(x0, y0):
                continue
            out.append(Patch(rec.path, x0, y0, s, 1, boxes_in_patch(rec.boxes, x0, y0, s), rec.source))
            made += 1
    # negatives: a fraction `hard_frac` are "hard" = the most glare/edge-rich of `hard_cands`
    # valid candidates (e.g. the bright dish rim), so the model cannot use them as a cue.
    def valid_neg():
        for _ in range(tries):
            x0, y0 = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
            if not _overlaps_any(x0, y0, s, rec.boxes, neg_margin) and fov_ok(x0, y0):
                return x0, y0
        return None

    for _ in range(n_neg):
        if hard is not None and rng.random() < hard_frac:
            cands = [c for c in (valid_neg() for _ in range(hard_cands)) if c is not None]
            if not cands:
                continue
            x0, y0 = max(cands, key=lambda c: float(hard[c[1]:c[1] + s, c[0]:c[0] + s].mean()))
        else:
            c = valid_neg()
            if c is None:
                continue
            x0, y0 = c
        out.append(Patch(rec.path, x0, y0, s, 0, np.zeros((0, 4), np.float32), rec.source))
    return out


def build_patch_set(records: list[ImageRecord], seed: int, patch: int, n_pos: int, n_neg: int,
                    fov_min: float = 0.0, neg_margin: int = 8, dark_thr: int = 12,
                    balance: bool = True, hard_frac: float = 0.0, hard_cands: int = 16,
                    glare_thr: int = 150) -> list[Patch]:
    rng = np.random.default_rng(seed)
    patches: list[Patch] = []
    for r in records:
        fov = cached_fov(r.path, dark_thr) if fov_min > 0 else None
        hard = cached_hardness(r.path, glare_thr) if hard_frac > 0 else None
        patches += sample_patches(r, rng, patch, n_pos, n_neg, fov, fov_min, neg_margin,
                                  hard=hard, hard_frac=hard_frac, hard_cands=hard_cands)
    if balance:
        pos = [p for p in patches if p.label == 1]
        neg = [p for p in patches if p.label == 0]
        k = min(len(pos), len(neg))
        if k:
            pos = [pos[i] for i in rng.choice(len(pos), k, replace=False)]
            neg = [neg[i] for i in rng.choice(len(neg), k, replace=False)]
            patches = pos + neg
    order = rng.permutation(len(patches))
    return [patches[i] for i in order]


def crop(img: np.ndarray, p: Patch) -> np.ndarray:
    return img[p.y0:p.y0 + p.size, p.x0:p.x0 + p.size]


# =========================================================================== sanity baseline
def variance_baseline(train: list[Patch], test: list[Patch]) -> dict:
    """If grey-level std alone separates the classes, the benchmark is too easy."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    def feats(ps):
        cache, f = {}, []
        for p in ps:
            if p.path not in cache:
                cache = {p.path: cv2.cvtColor(imread(p.path), cv2.COLOR_BGR2GRAY)}
            f.append(float(crop(cache[p.path], p).std()))
        return np.asarray(f), np.asarray([p.label for p in ps])

    ftr, ytr = feats(train)
    fte, yte = feats(test)
    best_t, best_acc, sign = 0.0, 0.0, 1
    for sg in (1, -1):
        for t in np.unique(ftr):
            acc = ((sg * ftr > sg * t) == ytr).mean()
            if acc > best_acc:
                best_t, best_acc, sign = t, acc, sg
    pred = (sign * fte > sign * best_t).astype(int)
    auc = roc_auc_score(yte, sign * fte) if len(set(yte)) == 2 else float("nan")
    return {"threshold": float(best_t), "direction": "std>t" if sign == 1 else "std<t",
            "train_acc": float(best_acc), "test_acc": float(accuracy_score(yte, pred)), "test_auc": float(auc)}


# =========================================================================== CAM -> boxes
def cam_to_boxes(cam: np.ndarray, tau: float, min_box: int = 6, fov: np.ndarray | None = None) -> np.ndarray:
    """Threshold a [0,1] CAM, split into connected components, return (N,5) xyxy+score
    where score = mean CAM inside the component."""
    mask = (cam > tau)
    if fov is not None:
        mask &= fov
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_box or h < min_box:
            continue
        score = float(cam[lab == i].mean())
        out.append([x, y, x + w, y + h, score])
    out = np.asarray(out, np.float32).reshape(-1, 5)
    return out[np.argsort(-out[:, 4])] if len(out) else out


def normalise_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo + 1e-8)


# =========================================================================== metrics
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def _match(pred: np.ndarray, gt: np.ndarray, thr: float) -> np.ndarray:
    """Greedy (score-ordered) one-to-one matching; returns TP flag per prediction."""
    tp = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    ious = iou_matrix(pred[:, :4], gt)
    used = np.zeros(len(gt), bool)
    for i in np.argsort(-pred[:, 4]):
        cand = np.where(~used)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= thr:
            tp[i], used[j] = True, True
    return tp


def average_precision(preds: list[np.ndarray], gts: list[np.ndarray], thr: float = 0.5) -> float:
    """VOC/COCO-style all-point interpolated AP over a set of images."""
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return float("nan")
    scores, flags = [], []
    for p, g in zip(preds, gts):
        if len(p):
            scores.append(p[:, 4])
            flags.append(_match(p, g, thr))
    if not scores:
        return 0.0
    scores, flags = np.concatenate(scores), np.concatenate(flags)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(flags[order])
    fp = np.cumsum(~flags[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def box_coverage(gts: np.ndarray, shape: tuple[int, int]) -> float:
    """Fraction of the image covered by the union of GT boxes = chance level of the pointing game."""
    m = np.zeros(shape, bool)
    for b in np.asarray(gts).astype(int):
        m[max(b[1], 0):b[3], max(b[0], 0):b[2]] = True
    return float(m.mean())


def localisation_report(preds: list[np.ndarray], gts: list[np.ndarray], peaks: list[tuple[int, int]],
                        shapes: list[tuple[int, int]] | None = None) -> dict:
    """preds: per image (N,5) boxes; gts: per image (M,4); peaks: per image (x,y) of CAM maximum;
    shapes: per image (H,W), used to report the pointing-game chance level."""
    idx = [i for i, g in enumerate(gts) if len(g)]
    if not idx:
        return {"images_with_gt": 0}
    corloc = np.mean([bool(len(preds[i])) and (iou_matrix(preds[i][:, :4], gts[i]) >= 0.5).any() for i in idx])
    point = np.mean([any(g[0] <= peaks[i][0] <= g[2] and g[1] <= peaks[i][1] <= g[3] for g in gts[i]) for i in idx])
    tp = sum(int(_match(preds[i], gts[i], 0.5).sum()) for i in idx)
    n_pred = sum(len(preds[i]) for i in idx)
    n_gt = sum(len(gts[i]) for i in idx)
    ap50 = average_precision([preds[i] for i in idx], [gts[i] for i in idx], 0.5)
    ap_range = float(np.nanmean([average_precision([preds[i] for i in idx], [gts[i] for i in idx], t)
                                 for t in np.arange(0.5, 0.96, 0.05)]))
    chance = float(np.mean([box_coverage(gts[i], shapes[i]) for i in idx])) if shapes else float("nan")
    return {"images_with_gt": len(idx), "gt_boxes": n_gt, "pred_boxes": n_pred,
            "corloc": float(corloc), "pointing_game": float(point), "pointing_game_chance": chance,
            "precision@0.5": tp / max(n_pred, 1), "recall@0.5": tp / max(n_gt, 1),
            "ap50": ap50, "ap50_95": ap_range}


# =========================================================================== audit
def audit_archive(root: str, out_dir: str, dark_thr: int = 12, n_examples: int = 6, seed: int = 0) -> dict:
    """Checks the dataset before any training and writes audit.json + diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    data, logs = load_archive(root)
    report = {"root": os.path.abspath(root), "splits": logs}

    hashes = defaultdict(list)
    sizes = Counter()
    fov_frac = []
    for split, recs in data.items():
        for r in recs:
            with open(r.path, "rb") as f:
                hashes[hashlib.md5(f.read()).hexdigest()].append(f"{split}/{r.name}")
            img = imread(r.path)
            sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1
            if (img.shape[1], img.shape[0]) != (r.width, r.height):
                report.setdefault("csv_size_mismatch", []).append(f"{split}/{r.name}")
            fov_frac.append(float(field_of_view(img, dark_thr).mean()))
    dups = {h: v for h, v in hashes.items() if len(v) > 1}
    report["image_sizes"] = dict(sizes)
    report["exact_duplicate_groups"] = len(dups)
    report["exact_duplicate_examples"] = list(dups.values())[:10]
    report["field_of_view_fraction"] = {"mean": float(np.mean(fov_frac)), "min": float(np.min(fov_frac)),
                                        "max": float(np.max(fov_frac))}

    per_split = {}
    all_wh = []
    for split, recs in data.items():
        src = Counter(r.source for r in recs)
        nb = [len(r.boxes) for r in recs]
        wh = np.concatenate([np.c_[r.boxes[:, 2] - r.boxes[:, 0], r.boxes[:, 3] - r.boxes[:, 1]]
                             for r in recs if len(r.boxes)] or [np.zeros((0, 2))])
        all_wh.append(wh)
        per_split[split] = {"images": len(recs), "unique_sources": len(src),
                            "max_copies_per_source": max(src.values()) if src else 0,
                            "boxes_per_image_mean": float(np.mean(nb)) if nb else 0.0,
                            "boxes_per_image_max": int(max(nb)) if nb else 0,
                            "box_w_median_px": float(np.median(wh[:, 0])) if len(wh) else 0.0,
                            "box_h_median_px": float(np.median(wh[:, 1])) if len(wh) else 0.0,
                            "box_side_p5_p95_px": [float(np.percentile(wh.min(1), 5)) if len(wh) else 0.0,
                                                   float(np.percentile(wh.max(1), 95)) if len(wh) else 0.0]}
    report["per_split"] = per_split
    split_src = {k: {r.source for r in v} for k, v in data.items()}
    report["source_overlap_between_folders"] = {f"{a}&{b}": len(split_src[a] & split_src[b])
                                                for a in split_src for b in split_src if a < b}

    warnings = []
    if report["exact_duplicate_groups"]:
        warnings.append(f"{report['exact_duplicate_groups']} groups of byte-identical images.")
    for k, v in report["source_overlap_between_folders"].items():
        if v:
            warnings.append(f"{v} source images appear in both {k} -> they will be removed from val/test.")
    for split, lg in logs.items():
        if lg["unannotated_images_on_disk"]:
            warnings.append(f"{split}: {lg['unannotated_images_on_disk']} images have no annotation rows "
                            f"(excluded; they may be true negatives or missing labels).")
        if lg["boxes_dropped_too_small"]:
            warnings.append(f"{split}: {lg['boxes_dropped_too_small']} degenerate boxes (<3 px) dropped.")
    report["warnings"] = warnings

    # figures: box-size histogram and annotated examples with field-of-view outline
    wh = np.concatenate(all_wh) if all_wh else np.zeros((0, 2))
    if len(wh):
        plt.figure(figsize=(6, 4))
        plt.hist(np.sqrt(wh[:, 0] * wh[:, 1]), bins=40, color="#3a6ea5")
        plt.xlabel("Box size  sqrt(w*h)  [px]")
        plt.ylabel("Count")
        plt.title("Annotated particle sizes")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "audit_box_sizes.png"), dpi=150)
        plt.close()
    rng = np.random.default_rng(seed)
    pool = [r for v in data.values() for r in v]
    pick = [pool[i] for i in rng.choice(len(pool), min(n_examples, len(pool)), replace=False)]
    tiles = []
    for r in pick:
        img = imread(r.path)
        fov = field_of_view(img, dark_thr).astype(np.uint8)
        cnts, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 200, 0), 2)
        for b in r.boxes.astype(int):
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        tiles.append(cv2.resize(img, (360, 360)))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imwrite(os.path.join(out_dir, "audit_examples.jpg"), np.vstack(rows))

    with open(os.path.join(out_dir, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report
