# Microplastic weakly supervised localisation (`archive` version)

This pipeline replaces the original single script. It reads your Roboflow export directly:

```
archive/
  train/  _annotations.csv  + *.jpg      (filename,width,height,class,xmin,ymin,xmax,ymax)
  valid/  _annotations.csv  + *.jpg
  test/   _annotations.csv  + *.jpg      (optional; used if present)
```

| File | Purpose |
|---|---|
| `mp_data.py` | Reads the CSVs, audits the data, keeps Roboflow copies together, splits the data, samples patches, computes metrics. Needs no PyTorch. |
| `mp_wsol.py` | Model, Helmholtz/L2/TV/Laplacian regularisers, training, CAM/Eigen-CAM, evaluation, ablation, prediction and ONNX export. |
| `test_mp_data.py` | Self-test on a synthetic archive: `python test_mp_data.py` |
| `requirements.txt` | `pip install -r requirements.txt` |

## How the data are used (and why)

1. **No fake "clear water" images.** Both classes are real 128 × 128 crops from your dish images.
   - **Positive** = the crop fully contains at least one annotated particle.
   - **Negative** = the crop has no annotated particle within 8 px.
2. **Hard negatives.** Half of the background patches are chosen from the most glare- and edge-rich areas, such as the dish rim, so "bright rim" cannot be used as a cue for plastic. (A field-of-view filter for microscope images exists but is off by default.)
3. **Training sees only the patch label** (plastic / no plastic). The boxes in `_annotations.csv` are used **only to score localisation**: CorLoc, pointing game, precision/recall at IoU 0.5, AP@0.5 and AP@[.5:.95].
4. **Leakage control.** Roboflow names its augmented copies `X_jpg.rf.<hash>.jpg`, and every copy of one original photo stays in the same split.
   - The `valid` folder is held out as the **test** set.
   - The validation set is a group split of `train`.
   - Any original photo found in both `train` and `valid` is removed from the test set.
   - The split is saved to `split_manifest.json` in each run folder.
5. **Sanity baseline.** A grey-level-variance threshold is evaluated alongside the model. If it scores almost as well as the CNN, the benchmark is still too easy.

## Version 2: what changed after your first run

Your first run (v1) learned almost nothing: training accuracy stayed at ~55%, F1 = 0.14, and the heatmaps lit up the Petri-dish rim. The causes and fixes:

| Problem in v1 | Fix in v2 |
|---|---|
| The field-of-view filter assumed microscope images with black surroundings. In your dish photos it threw away the dark dish interior, so training patches piled up around the bright rim. | Filter **off** by default (`--fov-min 0 --fov-filter false`). |
| The model learned that "rim" means plastic. | Half of the negatives are **hard negatives**: background patches with the most glare and edges, i.e. the rim (`--hard-frac 0.5`). |
| It predicted "background" for almost everything (AUC 0.75, but recall 0.08). | The plastic/background cut-off is **chosen on the validation set** (best MCC), and early stopping uses validation AUC. |
| Small particles were averaged away by global average pooling. | **Top-k pooling** (`--pool topk --topk-frac 0.05`) and smaller patches (`--patch 128`). |
| The 3-conv network could not even fit the training data. | BatchNorm plus 2 extra 3×3 convs (receptive field ≈ 38 px, about 0.37 M parameters). |
| The Helmholtz term was about 0.002 of the loss and could be reduced just by shrinking the features. | The penalty is **divided by the feature energy** (`--reg-norm true`), so it shapes the spectrum and cannot be avoided by shrinking. |
| The pointing-game score had no reference point. | The chance level (fraction of the image covered by boxes) is now reported. |

## Run plan (in this order; use NEW output folders, because v1 runs are not compatible)

PowerShell, from the code folder (adjust the `archive` path if needed):

```powershell
$ROOT = "..\..\archive"

# 0) self-test (should print ALL TESTS PASSED)
py -3.11 test_mp_data.py

# 1) one main run - CHECK IT before continuing
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\helm_s0 --reg helmholtz
```

**Check after step 1.** The printout should show train accuracy rising well above 0.6 and validation AUC above about 0.8. Open a few images in `runs2\helm_s0\overlays_test`. If accuracy is still near 0.55, or the heatmaps still sit on the rim, stop and send me the console output, `history.csv` and two overlays before running the rest.

```powershell
# 2) the key comparison for the paper: no penalty vs Helmholtz, 3 seeds each
py -3.11 mp_wsol.py ablate --data-root $ROOT --out runs2\abl --regs none,helmholtz --seeds 0,1,2

# 3) strong baseline: ImageNet-pretrained ResNet-18 (downloads weights once), 3 seeds
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\resnet_s0 --backbone resnet18 --pretrained true --reg none --seed 0
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\resnet_s1 --backbone resnet18 --pretrained true --reg none --seed 1
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\resnet_s2 --backbone resnet18 --pretrained true --reg none --seed 2

# 4) does kappa matter? (seed 0; the paper's argument depends on this)
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\kappa0.75 --reg helmholtz --kappa 0.75
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\kappa2.0 --reg helmholtz --kappa 2.0
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\laplacian --reg laplacian
py -3.11 mp_wsol.py train --data-root $ROOT --out runs2\l2 --reg l2

# 5) collect everything into two tables
py -3.11 mp_wsol.py summarize --runs "runs2\*" "runs2\abl\*" --out runs2\summary.csv
```

If time is short, steps 1, 2 and 5 are the minimum; step 3 is strongly recommended.

Optional extras:
- `--amp true` on an NVIDIA GPU.
- Extra seeds `--seeds 0,1,2,3,4` for tighter error bars.

## What to send back for the final manuscript

1. `runs2\summary.csv` and `runs2\summary_grouped.csv`
2. From the best Helmholtz run: `results.json`, `history.csv`, `training_curves.png`, `feature_spectrum.png`, `confusion_matrix.png`
3. The same `feature_spectrum.png` from a `none` run (e.g. `runs2\abl\none_s0`), for comparison
4. 4–6 images from `overlays_test`, including at least one good and one bad example
5. `runs\audit\audit.json` (dataset counts)
6. Your GPU/CPU model, and roughly how long one training run took

## What each run folder contains

| File | Content |
|---|---|
| `config.json`, `split_manifest.json`, `split_info.json`, `train_time_s.txt` | Exact settings, split and training time |
| `best.pt`, `last.pt` | Checkpoints; best is chosen by validation AUC |
| `history.csv`, `training_curves.png` | Training and validation cross-entropy, regulariser loss, accuracy, MCC |
| `results.json` | All test metrics (see below) |
| `confusion_matrix.png` | Test-patch confusion matrix |
| `overlays_test/` | Test images with the CAM heatmap, field of view (cyan), annotations (green) and predictions (red) |
| `feature_spectrum.png` | Power spectrum of the conv3 features, which tests what the Helmholtz term actually does |

`results.json` contains:

- `test_patch_classification`: accuracy, precision, recall, F1, MCC, AUC and confusion matrix at the threshold chosen on validation (`decision_threshold_from_val`). The `_argmax` version uses a fixed 0.5 cut-off.
- `sanity_variance_baseline`: the variance-threshold result.
- `localisation.gradcam` and `localisation.eigencam`: the threshold τ chosen on validation images, then CorLoc, pointing game, P/R, AP50 and AP50-95 on (a) full test images and (b) positive test patches.

`ablation/ablation_summary.csv` gives mean ± s.d. over seeds for every regulariser. Use it directly for the regulariser-comparison table (Table 2) and the λ/κ sensitivity table (Table 3) in the revised manuscript.

## Settings you may need to adjust

| Option | Default | When to change |
|---|---|---|
| `--patch` | 128 | Your median particle is about 28 px; 128 keeps one particle a reasonable share of the patch. |
| `--hard-frac` | 0.5 | Raise it to 0.7 if heatmaps still sit on the rim. |
| `--lam` | 0.1 | Try 0.03 or 0.3 if Helmholtz clearly helps or hurts. |
| `--fov-min`, `--fov-filter` | 0 / false | Only for true microscope images with black surroundings. |
| `--num-workers` | 0 | Raise it on Linux for speed; keep 0 on Windows if loading hangs. |
| `--amp` | false | Set `true` on an NVIDIA GPU for faster training. |
| `--batchnorm` | false | Try `true` if training is unstable. |

## Known limitations to state in the paper

- Negatives come from unannotated regions. Any particle the annotators missed becomes label noise. Inspect `overlays_test/` for red boxes on real but unlabelled particles.
- CAM thresholding tends to merge touching particles into one box, so recall on clusters will be below that of a supervised detector. Train YOLO11 on the same split (`split_manifest.json`) as the upper bound.
- Optical images cannot confirm polymer type; spectroscopic confirmation is still needed.
