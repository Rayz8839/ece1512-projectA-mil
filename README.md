# ECE1512 Project A: Multiple Instance Learning on whole-slide image features

Gated attention-based MIL (ABMIL) baseline on CAMELYON16, CAMELYON17 and BRACS using the supplied frozen 384-d patch
features, plus one research extension (Regularised ABMIL). Report: see `report/`.

> Adjust the folder names below if your repository differs from this layout.

## Repository layout

| Path | Content |
|---|---|
| `starter/` | Unmodified course starter code (data loader, configs, metric helpers) |
| `src/` | ABMIL model and training loop (`train_eval.py`) and the extension (`extension.py`) |
| `notebooks/` | Colab notebooks: baseline and extension |
| `results/` | Per-run histories, summaries, tables and figures |
| `docs/` | `proposals.md` (two proposals) and `reproducibility_checklist.md` (checklist for this repository) |
| `report/` | Written report |

## Data

The features are **not** stored in the repository (several GB). Download them from Zenodo
(https://zenodo.org/records/23130543) and place them as
`feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5`. The Colab notebooks download the HDF5 files from the Hugging Face
mirror `LeonYe/ECE1512-WSI-Feature`. The supplied C16 file has 398 slides, one fewer than the 399 official CAMELYON16 slides.

## How to run

1. Open the baseline notebook in Colab (GPU runtime, Tesla T4 was used), upload `Project_A_Starter_Code.zip` when asked and run all cells.
   It trains ABMIL and a mean-pooling ablation on the three datasets and writes the files in `results/`.
2. Open the extension notebook, upload the repository zip and run all cells. It re-runs the baseline, then the `full` variant on all three
   datasets and the `cb_only`, `drop_only` and `ent_only` ablations on C17 and BRACS.
3. Set `SEEDS = [42, 43, 44]` in the notebooks for mean +/- std (about three times longer).

## Protocol (identical for baseline and extension)

AdamW, learning rate 1e-4, weight decay 1e-5, batch size one slide, 30 epochs, seed 42, constant learning rate
(the `warmup_epoch` key in the supplied YAML is not used by our loop). The epoch with the highest validation AUC + macro-F1 is kept
(first one in case of ties) and the test set is evaluated once.

## Results (single run, seed 42)

| Dataset | Method | Acc (%) | AUC | Macro-F1 | Loss |
|---|---|---|---|---|---|
| C16 | ABMIL baseline | 88.37 | 0.9342 | 0.8668 | 0.2804 |
| C16 | Regularised ABMIL | 89.92 | 0.9250 | 0.8860 | 0.3297 |
| C17 | ABMIL baseline | 78.50 | 0.8391 | 0.5323 | 0.7047 |
| C17 | Regularised ABMIL | 78.00 | 0.8212 | 0.4841 | 0.7527 |
| BRACS | ABMIL baseline | 39.53 | 0.8042 | 0.3494 | 1.4265 |
| BRACS | Regularised ABMIL | 43.02 | 0.8000 | 0.3705 | 1.4000 |

Mean pooling instead of learned attention lowers AUC by 0.28 (C16), 0.19 (C17) and 0.14 (BRACS).

## Extension: Regularised ABMIL

The architecture is unchanged; only the training signal changes (no new parameters):

* **A1, class-weighted label smoothing** (effective-number weights, beta = 0.999, label smoothing 0.1). In the code this is `use_cb` / `cb_only`.
  With batch size one the class weights act only on the smoothing term, so this is **not** the class-balanced loss of Cui et al. (2019).
* **A2, random patch dropout** (keep 50% of the patches per training step; all patches at evaluation).
* **A3, attention-entropy penalty** (lambda = 0.1, normalised by log N).

The result is mixed and inconclusive: small accuracy and macro-F1 gains on C16 and BRACS, lower AUC on all three datasets, and worse
F1 on the smallest C17 classes. All numbers come from one seed and small test sets (one slide is 0.78, 0.5 and 1.16 accuracy points
on C16, C17 and BRACS), so no claim of improvement or degradation is made. See `docs/proposals.md` for both proposals
and the report for the full analysis.

## Reproducibility and licences

* The checklist for this repository is `docs/reproducibility_checklist.md`; the report has its own checklist in the report appendices.
* `starter/` is the unmodified course starter package; the data are third-party datasets (CAMELYON16/17, BRACS) and pre-extracted features
  distributed via Zenodo under their own licences. See the report references for citations.
