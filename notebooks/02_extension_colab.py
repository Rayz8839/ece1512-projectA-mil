# ===== ECE1512 Project A - Part 2: Regularised ABMIL (run in Google Colab, GPU runtime) =====
# Paste each "# %% CELL" block into its own Colab cell.

# %% CELL 1 - get the repo + starter code + features
import os, sys, subprocess, shutil
os.chdir('/content')
REPO_URL = "https://github.com/Rayz8839/ece1512-projectA-mil.git"   # <-- edit
if not os.path.exists('/content/ece1512-projectA-mil'):
    subprocess.run(["git", "clone", REPO_URL], check=True)
REPO = '/content/ece1512-projectA-mil'
PROJECT_DIR = f'{REPO}/starter'
subprocess.run(["pip", "install", "-q", "torchmetrics", "h5py", "pandas", "scikit-learn", "pyyaml", "tabulate", "huggingface_hub"], check=True)

# features (same source as the baseline notebook): Hugging Face dataset repo
from huggingface_hub import hf_hub_download
H5 = {}
for key in ["C16", "C17", "BRACS"]:
    H5[key] = hf_hub_download("LeonYe/ECE1512-WSI-Feature", f"feature/{key}/patch_feats_pretrain_medical_ssl.h5",
                              repo_type="dataset", local_dir="/content/datasets/hf")
    dst_dir = f"{PROJECT_DIR}/feature/{key}"; os.makedirs(dst_dir, exist_ok=True)
    dst = f"{dst_dir}/patch_feats_pretrain_medical_ssl.h5"
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(H5[key], dst)

os.chdir(PROJECT_DIR)
sys.path[:0] = [PROJECT_DIR, f"{REPO}/src"]
os.makedirs("results", exist_ok=True)

# %% CELL 2 - baseline (re-run for a fair comparison; same code/seed as Part 1)
import pandas as pd, numpy as np, matplotlib.pyplot as plt, torch
from train_eval import run_experiment, device
from extension import run_extension, VARIANTS
print("Device:", device)
SEEDS = [42]                      # use [42, 43, 44] for mean +/- std (3x longer)
KEYS = ["C16", "C17", "BRACS"]
BASE = {(k, s): run_experiment(k, seed=s, verbose=False) for k in KEYS for s in SEEDS}

# %% CELL 3 - full method on all datasets (the main result)
EXT = {}
for k in KEYS:
    for s in SEEDS:
        EXT[(k, s, "full")] = run_extension(k, seed=s, tag="full", verbose=False, **VARIANTS["full"])

# %% CELL 4 - component ablations (cb_only / drop_only / ent_only). Costly: run on C17 + BRACS first.
for k in ["C17", "BRACS"]:
    for name in ["cb_only", "drop_only", "ent_only"]:
        EXT[(k, SEEDS[0], name)] = run_extension(k, seed=SEEDS[0], tag=name, verbose=False, **VARIANTS[name])

# %% CELL 5 - comparison tables + figures
rows = []
for (k, s), r in BASE.items():
    rows.append({"Dataset": k, "Seed": s, "Method": "ABMIL baseline", "Acc (%)": r["test"]["acc"],
                 "AUC": r["test"]["auc"], "F1": r["test"]["f1"], "Loss": r["test"]["loss"], "Best ep": r["best_epoch"]})
for (k, s, name), r in EXT.items():
    rows.append({"Dataset": k, "Seed": s, "Method": name, "Acc (%)": r["test"]["acc"],
                 "AUC": r["test"]["auc"], "F1": r["test"]["f1"], "Loss": r["test"]["loss"], "Best ep": r["best_epoch"]})
cmp_df = pd.DataFrame(rows).sort_values(["Dataset", "Method"])
cmp_df.to_csv("results/extension_comparison.csv", index=False); display(cmp_df.round(4))

# mean +/- std table (meaningful only with >1 seed)
agg = cmp_df.groupby(["Dataset", "Method"])[["Acc (%)", "AUC", "F1", "Loss"]].agg(["mean", "std"]).round(4)
agg.to_csv("results/extension_comparison_meanstd.csv"); display(agg)

# attention sparsity: baseline vs full
att_rows = []
for k in KEYS:
    for label, d in [("baseline", BASE[(k, SEEDS[0])]["att_df"]), ("full", EXT[(k, SEEDS[0], "full")]["att_df"])]:
        att_rows.append({"Dataset": k, "Method": label, "norm. entropy": d.norm_entropy.mean(), "top-1% mass": d.top1pct_mass.mean()})
att_cmp = pd.DataFrame(att_rows); att_cmp.to_csv("results/extension_attention.csv", index=False); display(att_cmp.round(4))

# per-class F1 baseline vs full (where does the gain/loss come from?)
from sklearn.metrics import f1_score
pc_rows = []
for k in KEYS:
    for label, r in [("baseline", BASE[(k, SEEDS[0])]), ("full", EXT[(k, SEEDS[0], "full")])]:
        y, p = r["test_labels"].numpy(), r["test_probs"].argmax(1).numpy()
        f = f1_score(y, p, labels=list(range(int(r["test_probs"].shape[1]))), average=None, zero_division=0)
        pc_rows.append({"Dataset": k, "Method": label, "per-class F1": np.round(f, 3).tolist()})
display(pd.DataFrame(pc_rows)); pd.DataFrame(pc_rows).to_csv("results/extension_per_class_f1.csv", index=False)

# curves: val loss and val AUC, baseline vs full
fig, ax = plt.subplots(2, len(KEYS), figsize=(5.2 * len(KEYS), 7), squeeze=False)
for j, k in enumerate(KEYS):
    hb, he = BASE[(k, SEEDS[0])]["history"], EXT[(k, SEEDS[0], "full")]["history"]
    ax[0, j].plot(hb.val_loss, label="baseline"); ax[0, j].plot(he.val_loss, label="regularised")
    ax[0, j].set_title(f"{k} - val loss (plain CE)"); ax[0, j].legend(); ax[0, j].grid(alpha=.3)
    ax[1, j].plot(hb.val_auc, label="baseline"); ax[1, j].plot(he.val_auc, label="regularised")
    ax[1, j].set_title(f"{k} - val AUC"); ax[1, j].set_xlabel("Epoch"); ax[1, j].legend(); ax[1, j].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/extension_curves.png", dpi=200); plt.show()

# %% CELL 6 - download results (commit them to results/extension/ in your repo)
shutil.make_archive("/content/extension_results", "zip", "results")
from google.colab import files; files.download("/content/extension_results.zip")
