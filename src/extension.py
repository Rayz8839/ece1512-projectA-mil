"""Research extension: Regularised ABMIL.

Three independent, switchable training-time components (inference is unchanged):
  1. Class-balanced loss : effective-number class weights (Cui et al., CVPR 2019) + label smoothing
  2. Patch dropout       : keep a random fraction of each bag's patches at every training step
  3. Attention entropy   : lambda * normalised entropy of the attention distribution (sparser attention)

Same features, splits, seed, optimiser, epochs and model-selection rule as the baseline.
Validation/test loss is always plain cross-entropy so it is comparable to the baseline.
"""
import copy, math, time, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import load_config, seed_everything, infer_feature_dim
from models import ABMIL
from train_eval import evaluate, attention_stats, class_counts, CONFIG_FILES, device

VARIANTS = {
    "full":      dict(use_cb=True,  label_smooth=0.1, patch_keep=0.5, lam_ent=0.1),
    "cb_only":   dict(use_cb=True,  label_smooth=0.1, patch_keep=1.0, lam_ent=0.0),
    "drop_only": dict(use_cb=False, label_smooth=0.0, patch_keep=0.5, lam_ent=0.0),
    "ent_only":  dict(use_cb=False, label_smooth=0.0, patch_keep=1.0, lam_ent=0.1),
}


def make_class_weights(counts, n_class, beta=0.999):
    n = np.array([max(counts.get(c, 0), 1) for c in range(n_class)], dtype=np.float64)
    w = (1 - beta) / (1 - np.power(beta, n))
    w = w / w.sum() * n_class
    return torch.tensor(w, dtype=torch.float32)


def attn_entropy_loss(att):
    """Mean normalised entropy of attention [B,1,N]; 0 = peaked, 1 = uniform."""
    a = att.clamp_min(1e-12)
    n = a.shape[-1]
    ent = -(a * a.log()).sum(-1)
    return (ent / math.log(max(n, 2))).mean()


def run_extension(key, seed=42, use_cb=True, label_smooth=0.1, patch_keep=0.5, lam_ent=0.1,
                  epochs=None, tag="full", verbose=True, out_dir="results"):
    conf = load_config(CONFIG_FILES[key]); conf.seed = seed
    seed_everything(seed)
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
    conf.D_feat = infer_feature_dim(train_data)
    counts = class_counts(train_data, conf)

    g = torch.Generator().manual_seed(seed)
    pin = bool(conf.pin_memory) and torch.cuda.is_available()
    common = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker), pin_memory=pin, collate_fn=mil_collate_fn)
    train_loader = DataLoader(train_data, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)

    model = ABMIL(conf, pooling="attention").to(device)
    w = make_class_weights(counts, int(conf.n_class)).to(device) if use_cb else None
    train_crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smooth)
    eval_crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
    n_epochs = int(epochs if epochs is not None else conf.train_epoch)

    best_score, best_epoch, best_state, history = float("-inf"), -1, None, []
    t_start = time.time()
    for epoch in range(n_epochs):
        model.train(); tl, tc, tn = 0.0, 0, 0
        for data in train_loader:
            bag = data["input"].to(device, dtype=torch.float32)
            mask = data["mask"].to(device)
            labels = data["label"].to(device, dtype=torch.long)
            if patch_keep < 1.0:
                N = bag.shape[1]; k = max(1, int(patch_keep * N))
                idx = torch.randperm(N, device=device)[:k]
                bag, mask = bag[:, idx], mask[:, idx]
            optimizer.zero_grad()
            logits, att = model(bag, mask=mask, return_attention=True)
            loss = train_crit(logits, labels)
            if lam_ent > 0:
                loss = loss + lam_ent * attn_entropy_loss(att)
            loss.backward(); optimizer.step()
            tl += loss.item() * labels.shape[0]
            tc += (logits.argmax(1) == labels).sum().item(); tn += labels.shape[0]
        va = evaluate(model, eval_crit, val_loader, device, conf)
        score = float(np.nan_to_num(va["auc"]) + np.nan_to_num(va["f1"]))
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch + 1, "train_loss": tl / tn, "train_acc": 100 * tc / tn,
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_auc": va["auc"], "val_f1": va["f1"]})
        if verbose:
            print(f"[{key}|{tag}] ep {epoch+1:02d}/{n_epochs} train={tl/tn:.4f} val loss={va['loss']:.4f} "
                  f"auc={va['auc']:.4f} f1={va['f1']:.4f} best={best_epoch+1}")

    model.load_state_dict(best_state)
    test_m, test_probs, test_labels = evaluate(model, eval_crit, test_loader, device, conf, return_preds=True)
    print(f"[{key}|{tag}] best val epoch={best_epoch+1} | TEST acc={test_m['acc']:.2f} auc={test_m['auc']:.4f} "
          f"f1={test_m['f1']:.4f} loss={test_m['loss']:.4f}")
    att_df = attention_stats(model, test_data, device)
    pd.DataFrame(history).to_csv(f"{out_dir}/{key}_{tag}_seed{seed}_history.csv", index=False)
    with open(f"{out_dir}/{key}_{tag}_seed{seed}_summary.json", "w") as f:
        json.dump({"dataset": key, "seed": seed, "variant": tag, "best_epoch": best_epoch + 1,
                   "test": test_m, "train_time_s": time.time() - t_start,
                   "config": dict(use_cb=use_cb, label_smooth=label_smooth, patch_keep=patch_keep, lam_ent=lam_ent)}, f, indent=2)
    return {"key": key, "test": test_m, "history": pd.DataFrame(history), "best_epoch": best_epoch + 1,
            "att_df": att_df, "test_probs": test_probs, "test_labels": test_labels}
