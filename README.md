# ECE1512 Project A - Multiple-Instance Learning for Whole-Slide Images

ABMIL (Ilse et al., ICML 2018) baseline on CAMELYON16 / CAMELYON17 / BRACS using pre-extracted patch features,
plus a research extension (**Regularised ABMIL**: class-balanced loss + patch dropout + attention-entropy penalty).

**Report:** `report/report.pdf` | **Authors:** Syed Rayyan Albeez Ikramullah | **Course:** ECE1512, University of Toronto

## Repository layout
```
starter/      provided starter code (dataloader.py, utils.py, configs, dataset_csv) - NOT modified
src/          our code: models.py (ABMIL), train_eval.py (training/eval/baseline), extension.py (Regularised ABMIL)
notebooks/    01_baseline.ipynb (Colab baseline), 02_extension_colab.py (Colab cells for the extension)
results/
  baseline/   baseline tables, curves, confusion matrices, per-class reports, mean-pooling ablation
  extension/  extension comparison tables, attention statistics, curves
docs/         proposals.md (two research ideas), reproducibility_checklist.md (code artefact)
report/       final IEEE-style PDF report
```

## Data (not in the repo)
Pre-extracted features (384-d, frozen self-supervised backbone): Zenodo record 23130543 or Hugging Face
`LeonYe/ECE1512-WSI-Feature`. Place as `starter/feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5`
(the notebooks do this automatically).

## Reproduce (Google Colab, T4 GPU)
1. Baseline: open `notebooks/01_baseline.ipynb` and Run all (upload `Project_A_Starter_Code.zip` when asked).
2. Extension: in a new Colab, paste the cells from `notebooks/02_extension_colab.py` (set `REPO_URL` first) and Run all.
Settings: lr 1e-4, weight decay 1e-5, 30 epochs, batch = 1 slide, seed 42; model selection by validation AUROC + macro-F1; test evaluated once.

## Results (seed 42; fill the extension rows after running)
| Dataset | Method | Acc (%) | AUC | F1 | Loss |
|---|---|---|---|---|---|
| C16 | ABMIL baseline | 88.37 | 0.9342 | 0.8668 | 0.2804 |
| C16 | Regularised ABMIL | | | | |
| C17 | ABMIL baseline | 78.50 | 0.8391 | 0.5323 | 0.7047 |
| C17 | Regularised ABMIL | | | | |
| BRACS | ABMIL baseline | 39.53 | 0.8042 | 0.3494 | 1.4265 |
| BRACS | Regularised ABMIL | | | | |

Mean-pooling ablation (baseline): AUC 0.6584 / 0.6443 / 0.6616 on C16 / C17 / BRACS (see `results/baseline/aggregator_ablation.csv`).

## Attribution
`starter/` is course-provided code (unchanged). Everything in `src/`, `notebooks/`, `docs/` and `results/` is our own work.
Class-balanced weights follow Cui et al., CVPR 2019. ABMIL: Ilse, Tomczak, Welling, ICML 2018.
