# Research Extension Proposals (ECE1512 Project A, Part 3)

## Baseline observations that motivate both ideas
Source: `results/baseline/` (seed 42, 30 epochs, checkpoint chosen by validation AUC + F1).

| Dataset | Test Acc (%) | Test AUC | Test F1 | Observation |
|---|---|---|---|---|
| C16 (2 cls) | 88.37 | 0.9342 | 0.8668 | Val metrics saturate by epoch ~3; val loss ≈ 0 while test recall of class 1 is only 0.69 (val/test gap, 27 val slides) |
| C17 (4 cls) | 78.50 | 0.8391 | 0.5323 | Strong class imbalance; classes 1 and 2 have F1 0.24 / 0.19; train acc reaches ~99% while val loss stays noisy (~0.45-1.0) and val F1 fluctuates |
| BRACS (7 cls) | 39.53 | 0.8042 | 0.3494 | Best val epoch 6/30; val loss climbs from ~1.4 to ~2.6 while train acc reaches ~95% (strong overfitting) |

Mean-pooling ablation drops AUC by 0.19–0.28, so learned attention matters; but attention is fairly diffuse
(normalised entropy 0.72–0.84, top-1% mass 0.26–0.46).

## Idea A (implemented): Regularised ABMIL
**Motivation.** With only 242–389 training slides and ~10k patches per slide, ABMIL (165k parameters) memorises the training bags
(BRACS train acc ~95% vs val acc ~40%) and under-serves minority classes (C17 classes 1–2, BRACS classes 0, 2, 3, 4).

**Method (training-time only; the inference model is identical to ABMIL):**
1. *Class-balanced loss*: cross-entropy with effective-number weights w_c ∝ (1-β)/(1-β^{n_c}), β=0.999 (Cui et al. 2019), plus label smoothing 0.1. Rationale: rare classes get larger gradient share; smoothing limits over-confident fitting.
2. *Patch dropout*: at each step keep a random 50% of the patches of the bag. Rationale: bag-level augmentation; forces the slide embedding to be robust to missing instances, and halves compute. Attention is a softmax over remaining patches, so the slide embedding remains a weighted mean.
3. *Attention-entropy penalty*: L = L_CE + λ·H(α)/log N (λ=0.1). Rationale: baseline attention is diffuse; a mild entropy penalty encourages sparse, more interpretable attention on few diagnostic patches. The entropy is normalised by log N so λ is comparable across bag sizes.

**Design choices.** Validation/test loss stays plain CE; same splits, seed, optimiser, 30 epochs and model-selection rule as the baseline, so differences come only from the objective/augmentation. Each component is switchable for ablation (`cb_only`, `drop_only`, `ent_only`, `full`).

**Hypotheses.** (H1) Higher macro-F1 on C17 and BRACS (imbalanced); (H2) a smaller train/val gap; (H3) lower attention entropy / higher top-1% mass; (H4) little change on C16, which is already near saturation. A negative result (e.g. class weighting hurting AUC on tiny classes) is reported and analysed.

## Idea B (proposal only): Context-aware aggregation
**Motivation.** ABMIL scores each patch independently and sums them; diagnosis of BRACS sub-types (e.g. atypia vs DCIS) depends on spatial/neighbourhood context.
**Method.** Add a light self-attention (or Nyström/TransMIL-style) layer over patch embeddings before the gated attention pooling, with positional information if patch coordinates are available in the HDF5 files; subsample to ≤2k patches per bag (or use linear/Nyström attention) to keep the O(N²) cost feasible on a T4.
**Rationale/risks.** Captures inter-patch dependencies but adds parameters on very small datasets (overfitting risk), and cost grows with bag size. Would be compared against the same baseline and metrics.
