import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("figs.py").read().split("FIG =")[0])  # style + colors
S = json.load(open("/home/claude/results/11tumors/summary.json"))
LAB = {"LR": "LogReg-L2 (CV)", "LGBM": "LightGBM", "RF": "Random Forest", "SVM": "SVM-RBF", "MLP": "MLP",
       "Vote": "Voting SVM+kNN+DT", "kNN": "kNN", "DeepInsight": "DeepInsight-CNN", "CNN_corr": "CorrConvNet-1D (alone)",
       "Stack w/o CNN (classical only)": "Stack w/o CNN"}
FIG = "/home/claude/paper/figs"
# ---------- Option A: forest plot of paired differences ----------
T = S["tests_vs_proposed"]
ms = sorted([m for m in LAB if m in T["acc"]], key=lambda m: T["acc"][m]["delta"])
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, key, xl, sc in [(axs[0], "acc", "Accuracy difference (points)", 100), (axs[1], "auc", "Macro-AUC difference", 1)]:
    for i, m in enumerate(ms):
        t = T[key][m]; d = t["delta"] * sc; lo, hi = np.array(t["ci"]) * sc
        sig = t["p_holm"] < 0.05; excl = lo > 0 or hi < 0
        c = BLUE if sig else (ORANGE if excl else GREY)
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(d, i, "D" if sig else "o", ms=5, color=c, mec="white", mew=1)
    ax.axvline(0, color=INK, lw=0.8); ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(ms))); axs[0].set_yticklabels([LAB[m] for m in ms])
axs[0].text(0.02, 1.02, "← baseline better", transform=axs[0].transAxes, fontsize=6.5, color=MUTED)
axs[0].text(0.98, 1.02, "CorrConvNet-Stack better →", transform=axs[0].transAxes, fontsize=6.5, color=MUTED, ha="right")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=BLUE, marker="D", lw=2, label="Significant (Holm-adjusted p < 0.05)"),
     Line2D([], [], color=ORANGE, marker="o", lw=2, label="Bootstrap CI excludes 0 only"),
     Line2D([], [], color=GREY, marker="o", lw=2, label="Not significant")]
fig.legend(handles=h, frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.55, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FIG}/fig_main_forest.pdf"); fig.savefig("/home/claude/out/fig1_optionA_forest.png", dpi=200); plt.close(fig)

# ---------- Option B: accuracy–AUC scatter ----------
M = S["models"]; allm = ["CorrConvNet-Stack"] + list(LAB)
fig, ax = plt.subplots(figsize=(4.6, 3.2))
off = {"LR": (6, 4), "LGBM": (6, -9), "Stack w/o CNN (classical only)": (-8, 6), "RF": (6, -3), "MLP": (6, 2), "SVM": (6, -8),
       "Vote": (6, -3), "kNN": (6, 0), "DeepInsight": (6, 3), "CNN_corr": (6, 0), "CorrConvNet-Stack": (-10, 8)}
for m in allm:
    p, ci = M[m]["pooled"], M[m]["ci"]; me = m == "CorrConvNet-Stack"; c = BLUE if me else GREY
    ax.errorbar(100 * p["acc"], p["auc"], xerr=[[100 * (p["acc"] - ci["acc"][0])], [100 * (ci["acc"][1] - p["acc"])]],
                yerr=[[p["auc"] - ci["auc"][0]], [ci["auc"][1] - p["auc"]]], fmt="o", ms=6 if me else 4.5, color=c,
                ecolor=c, elinewidth=1 if me else 0.7, alpha=1 if me else 0.8, mec="white", mew=1, zorder=3 if me else 2)
    lab = "CorrConvNet-Stack (ours)" if me else LAB[m]
    ax.annotate(lab, (100 * p["acc"], p["auc"]), xytext=off.get(m, (6, 0)), textcoords="offset points", fontsize=6.3,
                color=INK if me else MUTED, ha="right" if off.get(m, (6, 0))[0] < 0 else "left", va="center", fontweight="bold" if me else "normal")
ax.set_xlabel("Accuracy (%)"); ax.set_ylabel("Macro AUC (OvR)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_main_scatter.pdf"); fig.savefig("/home/claude/out/fig1_optionB_scatter.png", dpi=200); plt.close(fig)
print("ok")
