# ECE1512 Project A - Multiple-Instance Learning for Whole-Slide Images

ABMIL (Ilse et al., ICML 2018) baseline on CAMELYON16 / CAMELYON17 / BRACS using pre-extracted patch features,
plus a research extension (**Regularised ABMIL**: class-balanced loss + patch dropout + attention-entropy penalty).

**Report:** `report/ECE1512_ProjectA_IEEE_Report.pdf` | **Author:** Syed Rayyan Albeez Ikramullah (M.Eng. Aerospace Engineering, University of Toronto) | **Course:** ECE1512

## Repository layout
```
starter/      provided starter code (dataloader.py, utils.py, configs, dataset_csv) - not modified
src/          our code: models.py (ABMIL), train_eval.py (training/eval/baseline), extension.py (Regularised ABMIL)
notebooks/    01_baseline_colab.ipynb (Colab baseline), 02_extension_colab.ipynb (Colab extension)
results/
  baseline/   baseline tables, curves, confusion matrices, per-class reports, mean-pooling ablation, per-epoch histories
  extension/  extension comparison tables, attention statistics, curves, per-epoch histories
docs/         proposals.md (two research ideas), reproducibility_checklist.md (code artefact)
report/       IEEE-style PDF report
```

## Data (not in the repo)
Pre-extracted features (384-d, frozen self-supervised backbone): Zenodo record 23130543 (the course starter's source).
Place as `starter/feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5`
(the notebooks do this automatically).

The supplied C16 file contains 398 slides (242 training, 27 validation, 129 test). The official CAMELYON16 split
has 270 training slides, so one training slide is absent from the supplied features.

## Reproduce (Google Colab, T4 GPU)
1. Baseline: open `notebooks/01_baseline_colab.ipynb` and Run all (upload the starter zip when asked).
2. Extension: open `notebooks/02_extension_colab.ipynb`, set `REPO_URL` to this repository, and Run all.

Settings: lr 1e-4, weight decay 1e-5, 30 epochs, batch = 1 slide, seed 42. The YAML configs contain a
`warmup_epoch` key that the training loop does not use, so the learning rate is constant.
Model selection: highest validation AUROC + macro-F1; the test set is evaluated once.

## Results (seed 42, single run per setting)
| Dataset | Method | Acc (%) | AUC | Macro-F1 | Loss |
|---|---|---|---|---|---|
| C16 | ABMIL baseline | 88.37 | 0.9342 | 0.8668 | 0.2804 |
| C16 | Regularised ABMIL | 89.92 | 0.9250 | 0.8860 | 0.3297 |
| C17 | ABMIL baseline | 78.50 | 0.8391 | 0.5323 | 0.7047 |
| C17 | Regularised ABMIL | 78.00 | 0.8212 | 0.4841 | 0.7527 |
| BRACS | ABMIL baseline | 39.53 | 0.8042 | 0.3494 | 1.4265 |
| BRACS | Regularised ABMIL | 43.02 | 0.8000 | 0.3705 | 1.4000 |

Source: `results/baseline/baseline_results_summary.csv`, `results/extension/extension_comparison.csv`.
Ablations (C17 and BRACS only) are in `results/extension/extension_comparison.csv`.

Mean-pooling ablation (baseline): AUC 0.6584 / 0.6443 / 0.6616 on C16 / C17 / BRACS (`results/baseline/aggregator_ablation.csv`).

Run times: the baseline archive (`results/baseline/`) and the extension archive (`results/extension/`) come from
separate Colab sessions. Compare run times only within a session (the extension archive includes its own baseline re-run).

## Attribution
`starter/` is course-provided code (unchanged). Everything in `src/`, `notebooks/`, `docs/`, `results/` and `report/` is our own work.
Class-balanced weights follow Cui et al., CVPR 2019. ABMIL: Ilse, Tomczak, Welling, ICML 2018.
