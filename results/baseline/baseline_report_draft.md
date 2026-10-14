# ABMIL Baseline Evaluation Summary

## 1. Key observations: model structure and training procedure

- **Input.** Each slide is a bag of pre-extracted patch features from a frozen self-supervised backbone (384-d per patch, variable number of patches per slide). Only the 3 modules below are trained.
- **Encoder (DimReduction).** Linear(384->256) + ReLU on every patch; 98,560 parameters.
- **Aggregator (gated attention).** a_i = w^T[tanh(V h_i) * sigmoid(U h_i)] with attention dim 128, softmax over the patches of a slide, slide embedding z = sum_i alpha_i h_i; 65,921 parameters.
- **Classifier.** One linear layer 256->n_classes on z; 514 parameters. Total trainable parameters: 164,995 (for 2 classes; the classifier grows with n_classes).
- **Training.** Cross-entropy loss, AdamW (lr=0.0001, weight decay=1e-05), batch size 1 slide per step, 30 epochs, no learning-rate schedule, dropout 0.0. Seed(s): [42].
- **Model selection / evaluation.** The epoch with the highest validation (macro-AUROC + macro-F1) is kept; the test set is evaluated once with that checkpoint. Metrics: accuracy, macro-AUROC, macro-F1, cross-entropy loss.
- **Splits.** CAMELYON16: official test slides held out, 10% of the rest for validation. CAMELYON17: centers >= 3 held out as test (center-shift), 10% of the remaining slides for validation. BRACS: split given in the CSV.

| Dataset   |   #Classes |   Train (slides) |   Val (slides) |   Test (slides) |
|:----------|-----------:|-----------------:|---------------:|----------------:|
| C16       |          2 |              242 |             27 |             129 |
| C17       |          4 |              269 |             30 |             200 |
| BRACS     |          7 |              389 |             62 |              86 |

## 2. Baseline performance

| Dataset   |   Test Acc (%) |   Test AUC |   Test F1 |   Test Loss |   #Seeds |
|:----------|---------------:|-----------:|----------:|------------:|---------:|
| C16       |          88.37 |     0.9342 |    0.8668 |      0.2804 |        1 |
| C17       |          78.5  |     0.8391 |    0.5323 |      0.7047 |        1 |
| BRACS     |          39.53 |     0.8042 |    0.3494 |      1.4265 |        1 |

Figures: `baseline_curves.png` (loss / accuracy / val AUC & F1 per epoch), `baseline_confusion_matrices.png`.

- **C16**: best validation epoch 3/30; final train loss 0.000 vs val loss 0.000 (gap -0.000); test acc 88.37%, AUC 0.9342, F1 0.8668. Class counts (train/val/test): {0: 142, 1: 100} / {0: 16, 1: 11} / {0: 80, 1: 49}.
- **C17**: best validation epoch 22/30; final train loss 0.033 vs val loss 0.801 (gap +0.768); test acc 78.50%, AUC 0.8391, F1 0.5323. Class counts (train/val/test): {0: 177, 1: 18, 2: 37, 3: 37} / {0: 20, 1: 2, 2: 4, 3: 4} / {0: 121, 1: 16, 2: 18, 3: 45}.
- **BRACS**: best validation epoch 6/30; final train loss 0.193 vs val loss 2.568 (gap +2.376); test acc 39.53%, AUC 0.8042, F1 0.3494. Class counts (train/val/test): {0: 24, 1: 119, 2: 54, 3: 24, 4: 28, 5: 40, 6: 100} / {0: 10, 1: 10, 2: 7, 3: 6, 4: 8, 5: 9, 6: 12} / {0: 6, 1: 16, 2: 9, 3: 11, 4: 12, 5: 12, 6: 20}.

## 3. What each component contributes

- **Encoder:** turns generic frozen features into a task-specific lower-dimensional space where patches that matter for the slide label become separable; it is applied identically to every patch.
- **Aggregator:** learns a per-patch importance score and pools the bag into one slide vector, so a small number of informative patches can dominate while the rest are down-weighted; it is also the only source of the interpretable attention map.
- **Classifier:** a linear decision layer on the slide embedding that maps it to class scores (softmax gives probabilities used for AUROC).

Attention statistics on test slides (normalised entropy 1 = uniform, 0 = a single patch):

| Dataset   |   mean #patches/slide |   mean normalised attention entropy (1=uniform) |   mean attention mass on top-1% patches |
|:----------|----------------------:|------------------------------------------------:|----------------------------------------:|
| C16       |              12513.4  |                                          0.8378 |                                  0.2627 |
| C17       |              10575    |                                          0.7195 |                                  0.3845 |
| BRACS     |               9042.13 |                                          0.7361 |                                  0.4554 |

Aggregator ablation (same encoder/classifier, uniform mean pooling instead of learned attention, same split and seed):

| Dataset   |   ABMIL AUC |   MeanPool AUC |   dAUC |   ABMIL F1 |   MeanPool F1 |    dF1 |   ABMIL Acc (%) |   MeanPool Acc (%) |
|:----------|------------:|---------------:|-------:|-----------:|--------------:|-------:|----------------:|-------------------:|
| C16       |      0.9342 |         0.6584 | 0.2758 |     0.8668 |        0.5883 | 0.2785 |         88.3721 |            67.4419 |
| C17       |      0.8391 |         0.6443 | 0.1948 |     0.5323 |        0.1885 | 0.3439 |         78.5    |            60.5    |
| BRACS     |      0.8042 |         0.6616 | 0.1427 |     0.3494 |        0.2103 | 0.1391 |         39.5349 |            33.7209 |

- **C16**: dAUC = +0.2758, dF1 = +0.2785 -> learned attention helped in this run (single seed; interpret small differences with care).
- **C17**: dAUC = +0.1948, dF1 = +0.3439 -> learned attention helped in this run (single seed; interpret small differences with care).
- **BRACS**: dAUC = +0.1427, dF1 = +0.1391 -> learned attention helped in this run (single seed; interpret small differences with care).

## 4. Caveats

- Test sets are small and some classes have few slides, so macro-F1/AUC are noisy; report mean +/- std over several seeds when possible.
- Attention weights show where the model looks, not causal evidence of what drives the prediction.
- The same features, splits and seed must be reused for the research extension so the comparison with this baseline is fair.