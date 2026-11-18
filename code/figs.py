import json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY, INK, MUTED, GRID = "#8a8a85", "#1f1f1d", "#5f5e5a", "#e1e0d9"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "savefig.dpi": 300, "savefig.bbox": "tight"})
FIG = "/home/claude/paper/figs"
S = {d: json.load(open(f"/home/claude/results/{d}/summary.json")) for d in ["11tumors", "leukemia"]}
X = pickle.load(open("/home/claude/results/11tumors/xai_raw.pkl", "rb"))
LAB = {"CorrConvNet-Stack": "CorrConvNet-Stack (ours)", "LR": "LogReg-L2 (CV)", "SVM": "SVM-RBF", "RF": "Random Forest",
       "kNN": "kNN", "MLP": "MLP", "LGBM": "LightGBM", "Vote": "Voting SVM+kNN+DT", "DeepInsight": "DeepInsight-CNN",
       "CNN_corr": "CorrConvNet-1D (alone)", "Stack w/o CNN (classical only)": "Stack w/o CNN"}

# 1. main comparison with 95% CIs
s = S["11tumors"]["models"]
ms = sorted(LAB, key=lambda m: s[m]["pooled"]["acc"])
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for ax, key, nm in [(axs[0], "acc", "Accuracy"), (axs[1], "auc", "Macro AUC (OvR)")]:
    for i, m in enumerate(ms):
        v, (lo, hi) = s[m]["pooled"][key], s[m]["ci"][key]
        c = BLUE if m == "CorrConvNet-Stack" else GREY
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(v, i, "o", ms=5, color=c, mec="white", mew=1)
    ax.set_xlabel(nm + " (95% bootstrap CI)"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(ms))); axs[0].set_yticklabels([LAB[m] for m in ms])
fig.savefig(f"{FIG}/fig_main.pdf"); plt.close(fig)

# 2. reordering ablation
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
cols = {"corr": BLUE, "anova": ORANGE, "random": AQUA}
names = {"corr": "Correlation (A)", "anova": "ANOVA rank (B)", "random": "Random (C)"}
for ax, key, nm in [(axs[0], "acc", "CNN accuracy"), (axs[1], "auc", "CNN macro AUC")]:
    for j, d in enumerate(["11tumors", "leukemia"]):
        for k, o in enumerate(["corr", "anova", "random"]):
            mm = S[d]["models"][f"CNN_{o}"]; v, (lo, hi) = mm["pooled"][key], mm["ci"][key]
            xpos = j + (k - 1) * 0.22
            ax.plot([xpos, xpos], [lo, hi], color=cols[o], lw=2, solid_capstyle="round")
            ax.plot(xpos, v, "o", ms=5, color=cols[o], mec="white", mew=1, label=names[o] if j == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["11_Tumors", "Leukemia"]); ax.set_title(nm, fontsize=8, color=INK); ax.grid(axis="x", visible=False)
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
for j, d in enumerate(["11tumors", "leukemia"]):
    for k, o in enumerate(["corr", "anova", "random"]):
        axs[2].bar(j + (k - 1) * 0.25, S[d]["locality"][o], 0.23, color=cols[o])
axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["11_Tumors", "Leukemia"]); axs[2].set_title("Mean |r| of adjacent genes", fontsize=8, color=INK)
axs[2].grid(axis="x", visible=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FIG}/fig_reorder.pdf"); plt.close(fig)

# 3. confusion matrix
cm = np.array(S["11tumors"]["confusion"]); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(3.2, 2.9))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1); ax.grid(False)
for i in range(len(cm)):
    for j in range(len(cm)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5, color="white" if cmn[i, j] > 0.55 else INK)
ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm))); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
fig.savefig(f"{FIG}/fig_confusion.pdf"); plt.close(fig)

# 4. training curve
cv = np.array(X["curve"])
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.8))
axs[0].plot(cv[:, 0], color=BLUE, lw=1.5); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Training loss")
axs[1].plot(cv[:, 1], color=ORANGE, lw=1.5); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Held-out accuracy")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_curve.pdf"); plt.close(fig)

# 5. XAI example: first correctly classified sample of fold 0
fx = X["fold0"]; yt = S["11tumors"]["classes"]
y_all = pickle.load(open("/home/claude/results/11tumors/rep0.pkl", "rb"))["y"]
yte = y_all[X["te0"]]
i = int(np.where(fx["cls"] == yte)[0][0])
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw=dict(width_ratios=[1.6, 1]))
axs[0].fill_between(np.arange(len(fx["gradcam"][i])), fx["gradcam"][i], color=BLUE, alpha=0.25, lw=0)
axs[0].plot(fx["gradcam"][i], color=BLUE, lw=1)
axs[0].set_xlabel("Gene position after correlation reordering"); axs[0].set_ylabel("Grad-CAM++ (norm.)")
lw = fx["lime"][i]; top = np.argsort(-np.abs(lw))[:12][::-1]
axs[1].barh(range(12), lw[top], color=[BLUE if v > 0 else ORANGE for v in lw[top]], height=0.7)
axs[1].set_yticks(range(12)); axs[1].set_yticklabels([f"D{fx['cols'][t]+1}" for t in top], fontsize=6)
axs[1].axvline(0, color="#c3c2b7", lw=0.8); axs[1].set_xlabel("LIME weight"); axs[1].locator_params(axis="x", nbins=3); axs[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2)); axs[1].grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_xai_example.pdf"); plt.close(fig)
json.dump(dict(sample=int(X["te0"][i]), cls=int(yte[i]), top_gc=[f"D{fx['cols'][t]+1}" for t in np.argsort(-fx["gradcam"][i])[:10]],
               top_gc_val=[float(fx["gradcam"][i][t]) for t in np.argsort(-fx["gradcam"][i])[:10]],
               top_lime=[f"D{fx['cols'][t]+1}" for t in np.argsort(-np.abs(lw))[:10]],
               top_lime_val=[float(lw[t]) for t in np.argsort(-np.abs(lw))[:10]]), open("/home/claude/results/xai_example.json", "w"), indent=1)

# 6. deletion curves
ks = S["11tumors"]["xai"]["ks"]; xm = S["11tumors"]["xai"]["methods"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2))
for nm, c in [("GradCAM++", BLUE), ("GradxInput", ORANGE), ("LIME", AQUA), ("Random", GREY)]:
    axs[0].plot(ks, xm[nm]["del_curve"], "-o", color=c, ms=3, lw=1.5, label=nm.replace("GradxInput", "Grad$\\times$Input"))
axs[0].set_xscale("log"); axs[0].set_xlabel("Top-$k$ genes deleted"); axs[0].set_ylabel("p(predicted class)"); axs[0].legend(frameon=False, fontsize=6.5)
axs[0].set_title("Deletion, correlation-ordered CNN", fontsize=8, color=INK)
for o, c, lab in [("", BLUE, "Correlation (A)"), ("_anova", ORANGE, "ANOVA rank (B)"), ("_random", AQUA, "Random (C)")]:
    g, r = xm["GradCAM++" + o], xm["Random" + o]
    axs[1].plot(ks, np.array(r["del_curve"]) - np.array(g["del_curve"]), "-o", color=c, ms=3, lw=1.5, label=lab)
axs[1].axhline(0, color="#c3c2b7", lw=0.8)
axs[1].set_xscale("log"); axs[1].set_xlabel("Top-$k$ genes deleted"); axs[1].set_ylabel("Extra drop vs. random")
axs[1].set_title("Grad-CAM++ faithfulness by input order", fontsize=8, color=INK); axs[1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_faith.pdf"); plt.close(fig)
print("figs ok")
