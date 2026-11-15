"""Self-check for mp_data.py on a synthetic archive (no PyTorch needed).
Run:  python test_mp_data.py      (or: pytest test_mp_data.py)"""
import os, sys, csv, json, secrets, tempfile
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mp_data as D


def make_mock(root):
    rng = np.random.default_rng(1)
    def make(path_dir, name, nb):
        W, H = 563, 537
        img = np.zeros((H, W, 3), np.uint8)
        cv2.circle(img, (W // 2, H // 2), 250, (60, 40, 55), -1)
        img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
        cv2.circle(img, (W // 2, H // 2), 250, (230, 230, 230), 4)
        rows = []
        for _ in range(nb):
            while True:
                r = int(rng.integers(8, 30)); cx = int(rng.integers(120, 440)); cy = int(rng.integers(120, 420))
                if (cx - W // 2) ** 2 + (cy - H // 2) ** 2 < 200 ** 2:
                    break
            cv2.ellipse(img, (cx, cy), (r, int(r * 0.6)), int(rng.integers(0, 180)), 0, 360, (40, 40, 200), -1)
            rows.append([name, W, H, "Microplastic", cx - r, cy - r, cx + r, cy + r])
        cv2.imwrite(os.path.join(path_dir, name), img)
        return rows
    for split, n, off in (("train", 30, 0), ("valid", 8, 25)):
        d = os.path.join(root, split); os.makedirs(d, exist_ok=True); allrows = []
        for i in range(n):
            src = f"c--{i + off}-_jpg"
            for _ in range(3 if split == "train" else 1):
                allrows += make(d, f"{src}.rf.{secrets.token_hex(16)}.jpg", int(rng.integers(1, 6)))
        allrows.append([allrows[0][0], 563, 537, "Microplastic", 58, 52, 59, 53])   # degenerate box
        with open(os.path.join(d, "_annotations.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(D.REQUIRED_COLS); w.writerows(allrows)


def test_pipeline():
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "archive"); make_mock(root)
    rep = D.audit_archive(root, os.path.join(tmp, "audit"))
    assert rep["per_split"]["train"]["unique_sources"] == 30
    assert rep["source_overlap_between_folders"]["train&valid"] == 5
    assert D.source_id("c--188-_jpg.rf.3bd16057.jpg") == "c--188-_jpg"
    data, _ = D.load_archive(root)
    splits, info = D.make_splits(data, 0.15, 0.15, 0)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert not ({r.source for r in tr} & {r.source for r in va})
    assert not ({r.source for r in tr} & {r.source for r in te})
    P = D.build_patch_set(tr, 0, 192, 2, 2)
    by = {r.path: r for r in tr}
    for p in P:
        if p.label == 0:
            assert not D._overlaps_any(p.x0, p.y0, p.size, by[p.path].boxes, 7)
        else:
            assert len(p.boxes) >= 1 and (p.boxes >= -1e-3).all() and (p.boxes <= p.size + 1e-3).all()
    assert sum(p.label for p in P) * 2 == len(P)                       # balanced
    hard = D.build_patch_set(tr, 0, 128, 2, 2, hard_frac=1.0)
    easy = D.build_patch_set(tr, 0, 128, 2, 2, hard_frac=0.0)
    score = lambda ps: np.mean([D.cached_hardness(p.path)[p.y0:p.y0 + p.size, p.x0:p.x0 + p.size].mean()
                                for p in ps if p.label == 0])
    assert score(hard) > score(easy)                                     # hard negatives sit on glare/edges
    for p in hard:
        if p.label == 0:
            assert not D._overlaps_any(p.x0, p.y0, p.size, by[p.path].boxes, 7)
    g = [np.array([[10, 10, 50, 50], [100, 100, 140, 150]], np.float32)]
    assert abs(D.average_precision([np.c_[g[0], [0.9, 0.8]].astype(np.float32)], g) - 1) < 1e-6
    half = [np.array([[10, 10, 50, 50, 0.9], [300, 300, 340, 350, 0.95]], np.float32)]
    assert abs(D.average_precision(half, g) - 0.25) < 1e-6
    cam = np.zeros((200, 200)); cam[20:60, 20:60] = 1; cam[100:150, 120:170] = .7
    assert len(D.cam_to_boxes(cam, .45)) == 2
    print("split info:", json.dumps({k: v for k, v in info.items() if k.endswith(("images", "boxes"))}))
    print("variance baseline:", D.variance_baseline(P, D.build_patch_set(te, 1, 192, 2, 2)))


if __name__ == "__main__":
    test_pipeline()
    print("ALL TESTS PASSED")
