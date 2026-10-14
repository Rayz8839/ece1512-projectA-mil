# ECE1512 Project A - Multiple-Instance Learning for Whole-Slide Images

ABMIL (Ilse et al., ICML 2018) baseline on CAMELYON16 / CAMELYON17 / BRACS using pre-extracted patch features,
plus a research extension (**Regularised ABMIL**: effective-number class weights with label smoothing, random patch dropout, attention-entropy penalty).

**Reports:** `report/ABMIL_Baseline_Report_IEEE_final.pdf` (Part B baseline), `report/PartC_Research_Extensions_Report_FINAL.pdf` (Part C extension)  
**Author:** Syed Rayyan Albeez Ikramullah | **Course:** ECE1512, University of Toronto

## Repository layout
```
starter/      provided starter code (dataloader.py, utils.py, configs, dataset_csv) - NOT modified
src/          our code: models.py (ABMIL), train_eval.py (training/eval/baseline), extension.py (Regularised ABMIL)
notebooks/    01_baseline_colab.ipynb (Colab baseline), 02_extension_colab.py (Colab cells for the extension)
results/
  baseline/   baseline tables, curves, confusion matrices, per-class reports, mean-pooling ablation
  extension/  extension comparison tables, ablations, attention statistics, curves
docs/         proposals.md (two research ideas), reproducibility_checklist.md (code artefact)
report/       final IEEE-style PDF reports (Part B and Part C)
```

## Data (not in the repo)
Pre-extracted features (384-d, frozen self-supervised backbone): Zenodo record 23130543 or Hugging Face
`LeonYe/ECE1512-WSI-Feature`. Place as `starter/feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5`
(the notebooks do this automatically).

## Reproduce (Google Colab, T4 GPU)
1. **Baseline:** open `notebooks/01_baseline_colab.ipynb` in Colab and Run all (upload `Project_A_Starter_Code.zip` when asked).
2. **Extension:** in a new Colab notebook, paste the cells from `notebooks/02_extension_colab.py` (each `# %% CELL` block into its own cell; `REPO_URL` is already set to this repository) and Run all.

Settings: lr 1e-4, weight decay 1e-5, 30 epochs, batch = 1 slide, seed 42; model selection by validation AUROC + macro-F1; test evaluated once.
Extension settings: label smoothing ε = 0.1, class-weight β = 0.999, patch keep ratio ρ = 0.5, entropy weight λ = 0.1 (`src/extension.py::VARIANTS`).

## Results (seed 42, single run per configuration)
| Dataset | Method | Acc (%) | AUC | F1 | Loss |
|---|---|---|---|---|---|
| C16 | ABMIL baseline | 88.37 | 0.9342 | 0.8668 | 0.2804 |
| C16 | Regularised ABMIL | 89.92 | 0.9250 | 0.8860 | 0.3297 |
| C17 | ABMIL baseline | 78.50 | 0.8391 | 0.5323 | 0.7047 |
| C17 | Regularised ABMIL | 78.00 | 0.8212 | 0.4841 | 0.7527 |
| BRACS | ABMIL baseline | 39.53 | 0.8042 | 0.3494 | 1.4265 |
| BRACS | Regularised ABMIL | 43.02 | 0.8000 | 0.3705 | 1.3999 |

Component ablation (seed 42; `results/extension/`):
| Dataset | Variant | Acc (%) | AUC | F1 | Loss |
|---|---|---|---|---|---|
| C17 | cb_only (A1) | 80.00 | 0.8232 | 0.4689 | 0.7075 |
| C17 | drop_only (A2) | 77.50 | 0.8250 | 0.5636 | 0.7804 |
| C17 | ent_only (A3) | 75.00 | 0.8154 | 0.5004 | 0.7478 |
| BRACS | cb_only (A1) | 39.53 | 0.7807 | 0.3494 | 1.4347 |
| BRACS | drop_only (A2) | 39.53 | 0.8026 | 0.3374 | 1.4170 |
| BRACS | ent_only (A3) | 39.53 | 0.8044 | 0.3494 | 1.4449 |

Mean-pooling ablation (baseline): AUC 0.6584 / 0.6443 / 0.6616 on C16 / C17 / BRACS (see `results/baseline/aggregator_ablation.csv`).

Results are from one seed and small test sets (86-200 slides); differences of a few points are not statistically established. See the Part C report, Sections V-VII.

## Attribution
`starter/` is course-provided code (unchanged). Everything in `src/`, `notebooks/`, `docs/`, `results/` and `report/` is our own work.
Class-balanced weights follow Cui et al., CVPR 2019; label smoothing follows Szegedy et al., CVPR 2016; ABMIL follows Ilse, Tomczak, Welling, ICML 2018.
