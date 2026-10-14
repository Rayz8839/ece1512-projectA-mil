# Machine Learning Reproducibility Checklist - CODE artefact (this GitHub repository)

(The written report has its own, separate checklist. Tick/complete before submission.)

- [ ✓] Dependencies specified (`requirements.txt`; Colab with a T4 GPU, PyTorch version printed in the notebooks)
- [ ✓] Training code provided (`src/train_eval.py`, `src/extension.py`) and evaluation code provided (same files)
- [ ✓] Pre-trained / pre-extracted features: source cited (Zenodo record 23130543, HF dataset `LeonYe/ECE1512-WSI-Feature`); not stored in the repo (4-7 GB each)
- [ ✓] Commands to reproduce each table/figure: see `README.md` ("Reproduce")
- [ ✓] Splits: provided by the starter `dataloader.py` and YAML configs (unchanged, in `starter/`); class counts in `results/baseline/splits_and_class_counts.csv`
- [ ✓] Hyper-parameters: from the starter YAMLs (lr 1e-4, wd 1e-5, 30 epochs, batch 1 slide); extension hyper-parameters in `src/extension.py::VARIANTS`
- [ ✓] Random seeds: 42 ; `seed_everything` from starter `utils.py`
- [ ✓] Model selection: best validation (AUROC + macro-F1) epoch; test evaluated once
- [ ✓] Metrics defined: accuracy, macro-AUROC, macro-F1 (starter `classification_metrics`), cross-entropy loss
- [  ] Number of runs / variance reported: ____ seeds
- [ ✓] Hardware / runtime: Google Colab Tesla T4; times in `results/*/*_summary.json`
- [ ✓] Provided starter code vs own code clearly separated (`starter/` vs `src/`)
- [ ✓] Licence / citation of third-party code and data included in README
