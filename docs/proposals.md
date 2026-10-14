# Research Extension Proposals (ECE1512 Project A, Part 3)

Both ideas keep the frozen 384-d features, the supplied splits and seed 42 used for the ABMIL baseline, so any
difference can be attributed to the idea and not to the data pipeline.

## Baseline observations that motivate the ideas (seed 42, 30 epochs, checkpoint chosen by validation AUC + macro-F1)

| Dataset | Test acc (%) | AUC | Macro-F1 | Observation |
|---|---|---|---|---|
| C16 (2 classes) | 88.37 | 0.9342 | 0.8668 | Validation metrics saturate by epoch 3; recall on class 1 is only 0.69 |
| C17 (4 classes) | 78.50 | 0.8391 | 0.5323 | Strong imbalance (177:18 in training); F1 of classes 1 and 2 is 0.24 and 0.19 |
| BRACS (7 classes) | 39.53 | 0.8042 | 0.3494 | Overfitting: training accuracy about 95%, validation loss rises from about 1.4 to 2.6 |

Replacing learned attention with mean pooling lowers test AUC by 0.28 (C16), 0.19 (C17) and 0.14 (BRACS), so the
attention aggregator matters. Attention is still fairly diffuse (normalised entropy 0.72-0.84).

## Idea A (implemented): Regularised ABMIL

Idea A keeps the ABMIL architecture unchanged (no new parameters) and only changes the training signal.
The three components can be switched on independently.

* **A1. Class-weighted label smoothing (effective-number weights).** Per-class weights
  w_c proportional to (1 - beta) / (1 - beta^n_c), beta = 0.999, rescaled to sum to the number of classes K, are passed
  to PyTorch's `CrossEntropyLoss` together with label smoothing epsilon = 0.1.
  With batch size one, the weight of the true class cancels from the hard-label term, so the weights act only on the
  smoothing term:
  `L_cls = (1 - eps) * CE(f(X), y) + eps / (K * w_y) * sum_c w_c * (-log p_c)`.
  **This is therefore not the class-balanced loss of Cui et al. (2019)**, which multiplies the whole per-sample loss by
  the class weight. In the code this component is enabled by `use_cb` and the ablation is called `cb_only`; the names are
  kept so that the saved result files stay reproducible.
* **A2. Random patch dropout.** At every training step a random 50% of the patches of the slide is kept (at least one);
  the attention softmax is taken over the kept patches. All patches are used at validation and test time.
* **A3. Attention-entropy penalty.** `L = L_cls + lambda * H(a) / log N`, lambda = 0.1, where `a` are the attention weights
  over the N kept patches. It favours peaked attention.

`full` uses A1 + A2 + A3. Ablations `cb_only`, `drop_only` and `ent_only` enable one component each (run on C17 and BRACS).

**Pre-stated hypotheses.**
H1: higher macro-F1 on the imbalanced C17 and BRACS tasks.
H2: a smaller train/validation gap.
H3: lower attention entropy and higher top-1% attention mass.
H4: little change on C16, which is close to saturation.
With one seed and test differences of one to three slides, none of these is claimed as confirmed or rejected;
negative results are reported as such.

## Idea B (proposal only): Context-aware aggregation

Insert one lightweight self-attention block (residual connection, 4 heads, 256-d) on the encoded patch features before
the gated attention, optionally with a TransMIL-style positional module if patch coordinates are available. A
zero-initialised residual gate makes the model start as the baseline. Full self-attention on N of about 10^4 patches
needs an N x N matrix (about 0.4 GB in float32 per head), too large for a T4 with gradients, so the planned options are
attention over a random subset of about 1-2k patches during training or Nystrom landmarks. The main risks are overfitting
with only 269-389 training slides and compute growing with bag size. Planned evaluation: same features, splits and seed;
report AUC, macro-F1 and parameter count against the baseline and against Idea A.
