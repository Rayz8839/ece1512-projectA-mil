"""Training / evaluation helpers and the baseline experiment runner.

Requires the starter folder (dataloader.py, utils.py) on sys.path and the
working directory set to the starter folder (config paths are relative).
"""
import copy, time, json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import build_HDF5_feat_dataset, mil_collate_fn, _labels_for_names
from utils import load_config, seed_everything, infer_feature_dim, classification_metrics
from models import ABMIL

CONFIG_FILES = {
    "C16":   "config/camelyon16_medical_ssl_config.yml",
    "C17":   "config/camelyon17_medical_ssl_config.yml",
    "BRACS": "config/bracs_medical_ssl_config.yml",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for data in data_loader:
        bag = data["input"].to(device, dtype=torch.float32)
        mask = data["mask"].to(device)
        labels = data["label"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += n
    return {"loss": total_loss / total_examples, "acc": 100.0 * total_correct / total_examples}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf, return_preds=False):
    model.eval()
    total_loss, total_examples = 0.0, 0
    probs_all, labels_all = [], []
    for data in data_loader:
        bag = data["input"].to(device, dtype=torch.float32)
        mask = data["mask"].to(device)
        labels = data["label"].to(device, dtype=torch.long)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.shape[0]
        total_examples += labels.shape[0]
        probs_all.append(F.softmax(logits, dim=1))
        labels_all.append(labels)
    probs_all = torch.cat(probs_all, 0)
    labels_all = torch.cat(labels_all, 0)
    try:
        metrics = classification_metrics(probs_all, labels_all, conf.n_class)
    except Exception as e:
        print("  [warn] metric computation failed:", e)
        metrics = {"auc": float("nan"), "f1": float("nan"),
                   "acc": float((probs_all.argmax(1) == labels_all).float().mean() * 100)}
    metrics["loss"] = total_loss / total_examples
    if return_preds:
        return metrics, probs_all.cpu(), labels_all.cpu()
    return metrics


@torch.no_grad()
def attention_stats(model, dataset, device):
    model.eval()
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        x = item["input"].unsqueeze(0).to(device, dtype=torch.float32)
        _, att = model(x, return_attention=True)
        a = att[0, 0].clamp_min(1e-12)
        n = a.numel()
        entropy = float(-(a * a.log()).sum())
        k = max(1, int(0.01 * n))
        rows.append({"slide_id": item["slide_id"], "label": item["label"], "n_patches": n,
                     "norm_entropy": entropy / float(np.log(n)) if n > 1 else 0.0,
                     "top1pct_mass": float(a.topk(k).values.sum())})
    return pd.DataFrame(rows)


def count_params(model):
    return {"encoder (DimReduction)": sum(p.numel() for p in model.dimreduction.parameters()),
            "aggregator (gated attention)": sum(p.numel() for p in model.attention.parameters()),
            "classifier (1 FC)": sum(p.numel() for p in model.classifier.parameters())}


def class_counts(ds, conf):
    labs = _labels_for_names(conf.data_dir, ds.data_names, conf)
    return {int(k): int(v) for k, v in sorted(pd.Series(labs).value_counts().items())}


def run_experiment(key, seed=42, pooling="attention", epochs=None, verbose=True, out_dir="results"):
    conf = load_config(CONFIG_FILES[key]); conf.seed = seed
    seed_everything(seed)
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
    conf.D_feat = infer_feature_dim(train_data)
    counts = {"train": class_counts(train_data, conf), "val": class_counts(val_data, conf),
              "test": class_counts(test_data, conf)}
    print(f"\n### {key} | seed {seed} | pooling={pooling} | splits {len(train_data)}/{len(val_data)}/{len(test_data)}")

    g = torch.Generator().manual_seed(seed)
    pin = bool(conf.pin_memory) and torch.cuda.is_available()
    common = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker), pin_memory=pin, collate_fn=mil_collate_fn)
    train_loader = DataLoader(train_data, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)

    model = ABMIL(conf, pooling=pooling).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
    n_epochs = int(epochs if epochs is not None else conf.train_epoch)

    best_score, best_epoch, best_state, history = float("-inf"), -1, None, []
    t_start = time.time()
    for epoch in range(n_epochs):
        tr = train_one_epoch(model, criterion, train_loader, optimizer, device)
        va = evaluate(model, criterion, val_loader, device, conf)
        score = float(np.nan_to_num(va["auc"]) + np.nan_to_num(va["f1"]))
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch + 1, "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_auc": va["auc"], "val_f1": va["f1"]})
        if verbose:
            print(f"Epoch {epoch+1:02d}/{n_epochs} train loss={tr['loss']:.4f} | val loss={va['loss']:.4f} "
                  f"auc={va['auc']:.3f} f1={va['f1']:.3f} | best={best_epoch+1}")
    model.load_state_dict(best_state)
    test_m, test_probs, test_labels = evaluate(model, criterion, test_loader, device, conf, return_preds=True)
    print(f"[{key}] best val epoch={best_epoch+1} | TEST acc={test_m['acc']:.2f} auc={test_m['auc']:.4f} "
          f"f1={test_m['f1']:.4f} loss={test_m['loss']:.4f}")
    tag = f"{key}_{pooling}_seed{seed}"
    pd.DataFrame(history).to_csv(f"{out_dir}/{tag}_history.csv", index=False)
    with open(f"{out_dir}/{tag}_summary.json", "w") as f:
        json.dump({"dataset": key, "seed": seed, "pooling": pooling, "best_epoch": best_epoch + 1,
                   "test": test_m, "train_time_s": time.time() - t_start, "class_counts": counts}, f, indent=2)
    att_df = attention_stats(model, test_data, device) if pooling == "attention" else None
    return {"key": key, "seed": seed, "conf": conf, "model": model, "best_epoch": best_epoch + 1,
            "history": pd.DataFrame(history), "test": test_m, "test_probs": test_probs,
            "test_labels": test_labels, "counts": counts, "att_df": att_df}
