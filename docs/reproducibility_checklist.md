# Machine Learning Reproducibility Checklist - CODE artefact (this GitHub repository)

The written report has its own, separate checklist (Appendix A of the report, Table XII). Items 11-15 of that checklist refer to this
repository; the answers below match the report's answers to those items.

- [x] Dependencies specified: `requirements.txt`, pinned to the versions in the Colab pip freeze (torch 2.11.0, h5py 3.14.0, pandas 2.2.3,
      scikit-learn 1.6.1, numpy 2.1.3, matplotlib 3.10.0, pyyaml 6.0.3, tabulate 0.9.0); `torchmetrics` is unpinned (version not recorded).
      Hardware: Google Colab, Tesla T4 GPU.
- [x] Training code provided: `src/train_eval.py` (baseline) and `src/extension.py` (Regularised ABMIL); evaluation code is in the same files.
- [x] Pre-trained models: the trained ABMIL checkpoints (seed 42) are in `results/baseline/*_best.pt` and `results/extension/`.
- [x] Pre-extracted features: source cited (Zenodo record 23130543); not stored in the repository (several GB each).
- [ ] Commands to reproduce each table/figure: not provided as one command per table. The two notebooks run end to end and write the files listed
      in `results/`; see the README section "How to run" (`README.md`, "How to run"). This item is marked not met, consistent with report Table XII item 15 (answer "No").
- [x] Splits: provided by the starter `dataloader.py` and YAML configs (unchanged, in `starter/`); class counts in
      `results/baseline/splits_and_class_counts.csv`.
- [x] Hyper-parameters: from the starter YAMLs (lr 1e-4, wd 1e-5, 30 epochs, batch 1 slide); extension hyper-parameters in
      `src/extension.py::VARIANTS`.
- [x] Random seeds: 42 (`seed_everything` from starter `utils.py`).
- [x] Model selection: best validation (AUROC + macro-F1) epoch; test evaluated once.
- [x] Metrics defined: accuracy, macro-AUROC, macro-F1 (starter `classification_metrics`), cross-entropy loss.
- [x] Number of runs: 1 seed (42) per setting for every reported result.
- [ ] Variance reported: not reported (one seed; stated as a limitation in the report; report Table XII item 19 answer "No").
- [x] Runtime: times in `results/*/*_summary.json` (`train_time_s`).
- [x] Provided starter code vs own code clearly separated (`starter/` vs `src/`).
- [x] Licence / citation of third-party code and data included in README.
