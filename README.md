# ECE1512 Project A: Multiple Instance Learning on whole-slide image features

Gated attention-based MIL (ABMIL) baseline on CAMELYON16, CAMELYON17 and BRACS using the supplied frozen 384-d patch
features, plus one research extension (Regularised ABMIL). Report: see `report/`.

## Repository layout

| Path | Content |
|---|---|
| `starter/` | Course starter code (data loader, configs, metric helpers), plus the completed student notebook. The starter files are unchanged except that its README is renamed `README_provided.md` so it does not clash with this README. |
| `src/` | ABMIL model and training loop (`train_eval.py`) and the extension (`extension.py`) |
| `notebooks/` | Colab notebooks: baseline and extension |
| `results/` | Per-run histories, summaries, tables, figures and trained checkpoints (`results/baseline/`: baseline; `results/extension/`: Regularised ABMIL and its ablations). `results/baseline_results.zip` and `results/extension_results.zip` are archives of those two folders, as produced by the notebooks' download cells. |
| `tools/` | `prepare_features.py`: copies the Zenodo feature archives to the paths the configs expect |
| `docs/` | `proposals.md` (two proposals) and `reproducibility_checklist.md` (checklist for this repository) |
| `report/` | Written report (PDF of the final submission) |

## Data

The features are **not** stored in the repository (several GB each).

**Option 1, Zenodo (the source named in the starter):** download the archive for each dataset from
https://zenodo.org/records/23130543, then run from the repository root:

```bash
python tools/prepare_features.py C16   path/to/<C16 archive>.zip
python tools/prepare_features.py C17   path/to/<C17 archive>.zip
python tools/prepare_features.py BRACS path/to/<BRACS archive>.zip
```

The script copies the HDF5 file to `starter/feature/<dataset>/patch_feats_pretrain_medical_ssl.h5`, which is the `data_dir`
in `starter/config/*.yml`. It accepts a `.zip` or an already-extracted `.h5`.

**Option 2, Hugging Face mirror (used by the Colab notebooks):** `LeonYe/ECE1512-WSI-Feature`. The Colab notebooks download
`feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5` from there, and the reported results were produced from those files.

The supplied C16 file has 398 slides, one fewer than the 399 official CAMELYON16 slides.

## How to run

0. Install the dependencies from the repository root: `pip install -r requirements.txt` (OpenSlide is needed only for the
   attention-visualization cell of the student notebook). Place the features as described in **Data**. The student TODO notebook is
   run separately, see **Starter TODO notebook** below.
1. Open the baseline notebook in Colab (GPU runtime, Tesla T4 was used), upload `Project_A_Starter_Code.zip` when asked and run all cells.
   It trains ABMIL and a mean-pooling ablation on the three datasets and writes the files to `starter/results/` inside Colab.
2. Open the extension notebook, upload this repository as a zip and run all cells. It re-runs the baseline, then the `full` variant on all three
   datasets and the `cb_only`, `drop_only` and `ent_only` ablations on C17 and BRACS. Its files are written to `starter/results/`
   inside Colab; copy them into this repository's `results/` folder.
3. There is no single command per table or figure: each notebook runs end to end and produces the files listed in `results/`.
4. `SEEDS = [42, 43, 44]` in the notebooks gives mean +/- std over three seeds (about three times longer). This option was
   **not run** for the reported results, which use one seed (report Sections X-E and XI-F).

## Starter TODO notebook (`starter/ECE1512_ProjectA_Student.ipynb`)

The course starter asks for `ECE1512_ProjectA_Student.ipynb` with every `TODO` cell completed. That completed notebook is included
in this repository at `starter/ECE1512_ProjectA_Student.ipynb`, next to the `dataloader.py` and `utils.py` it imports.

* **What it contains:** the DataLoaders (training shuffled, validation and test not), the encoder, gated-attention aggregator and classifier
  with their `forward()` code, the AdamW and cross-entropy setup, one training epoch, evaluation, validation-based checkpoint selection,
  a single final test evaluation, and the attention inspection cell.
* **Relation to the reported results:** the notebook is the guided implementation. The reported numbers come from `src/` and the two
  Colab notebooks. Both use the same features, splits, seed, optimizer and selection rule, but the student notebook has not been run in
  this repository, so its outputs are not part of the reported results.
* **Status:** every TODO is filled in and every code cell parses. It has not been executed end to end yet; run it once before relying on it.
* **How to run:** from the repository root, `pip install -r requirements.txt` and place the features (see **Data**). Then `cd starter`
  and open `ECE1512_ProjectA_Student.ipynb` in Jupyter, or upload the `starter/` folder to Colab. Paths are relative (`PROJECT_DIR = './'`),
  so run it from `starter/`. Set `DATASET` to `C16`, `C17` or `BRACS`. `NUM_EPOCHS` defaults to the YAML's 30 epochs; set it to 2 for a
  quick debug run.
* **Attention visualization (C16 only):** the last figure cell needs the WSI files `test_067.tif`, `test_068.tif` and `test_068.xml` in
  `starter/slide/` (from the Hugging Face dataset) and OpenSlide. Other cells do not need them.

## Protocol (identical for baseline and extension)

AdamW, learning rate 1e-4, weight decay 1e-5, batch size one slide, 30 epochs, seed 42, constant learning rate
(the `warmup_epoch` key in the supplied YAML is not used by our loop). The epoch with the highest validation AUC + macro-F1
is kept (first one in case of ties) and the test set is evaluated once.

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

* The checklist for this repository is [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md); the report has its own checklist (Appendix A of the report).
* `starter/` is the course starter package, unchanged except for the README rename and the added completed student notebook; the data are third-party datasets (CAMELYON16/17, BRACS) and pre-extracted features
  distributed via Zenodo under their own licences. See the report references for citations.
