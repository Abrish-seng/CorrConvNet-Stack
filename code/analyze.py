"""Aggregate all jobs for one dataset into results/<ds>/summary.json + figures."""
import warnings; warnings.filterwarnings("ignore")
import glob, json, pickle, sys, os
import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix, balanced_accuracy_score
from corrconv import meta_fit_predict

ds = sys.argv[1]
R = os.environ.get("CC_OUT", "/home/claude/results") + f"/{ds}"
FIG = f"/home/claude/paper/figs"
os.makedirs(FIG, exist_ok=True)
reps = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{R}/rep*.pkl"))]
perm = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{R}/perm*.pkl"))]
y = reps[0]["y"]; classes = reps[0]["classes"]; nc = len(classes); N = len(y)
B = 2000
rng = np.random.RandomState(0)

STACKS = {
    "CorrConvNet-Stack": (["LR", "SVM", "RF", "CNNs0_corr"], "LR"),
    "Stack w/o CNN (classical only)": (["LR", "SVM", "RF"], "LR"),
    "Stack w/o LR": (["SVM", "RF", "CNNs0_corr"], "LR"),
    "Stack w/o SVM": (["LR", "RF", "CNNs0_corr"], "LR"),
    "Stack w/o RF": (["LR", "SVM", "CNNs0_corr"], "LR"),
    "Stack, CNN on ANOVA order": (["LR", "SVM", "RF", "CNNs0_anova"], "LR"),
    "Stack, meta = average": (["LR", "SVM", "RF", "CNNs0_corr"], "avg"),
    "Stack, meta = LR (C=0.1)": (["LR", "SVM", "RF", "CNNs0_corr"], "LR_C0.1"),
    "Stack, meta = RF": (["LR", "SVM", "RF", "CNNs0_corr"], "RF"),
    "Stack, meta = Gaussian NB": (["LR", "SVM", "RF", "CNNs0_corr"], "NB"),
}


def stack_fold(fold, bases, meta, leaky=False):
    Mtr = np.hstack([fold["oof"][b + ("_leaky" if leaky else "")] for b in bases])
    Mte = np.hstack([fold["test"][b] for b in bases])
    return meta_fit_predict(meta, Mtr, fold["y_tr"], Mte, nc, len(bases))


def collect(rep):
    """-> dict model -> (pooled OOF probs (N,nc), list of per-fold (te, probs))"""
    yy = rep["y"]
    out = {}
    for fold in rep["folds"]:
        fold["y_tr"] = yy[fold["tr"]]
        P = dict(fold["test"])
        for nm, (b, m) in STACKS.items():
            if all(x in fold["oof"] for x in b):
                P[nm] = stack_fold(fold, b, m)
        if "LR_leaky" in fold["oof"]:
            P["Classical stack (leaky design)"] = stack_fold(fold, ["LR", "SVM", "RF"], "LR", leaky=True)
        for nm, p in P.items():
            out.setdefault(nm, [np.zeros((len(yy), nc)), []])
            out[nm][0][fold["te"]] = p
            out[nm][1].append((fold["te"], p))
    return out


def auc_ovr(yt, P):
    aucs = []
    for c in range(P.shape[1]):
        pos = yt == c; npos = pos.sum(); nneg = len(yt) - npos
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(P[:, c])
        aucs.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs))


def metrics(yt, P):
    yp = P.argmax(1)
    return dict(acc=float((yp == yt).mean()), bacc=float(balanced_accuracy_score(yt, yp)),
                f1=float(f1_score(yt, yp, average="macro", labels=range(nc), zero_division=0)),
                prec=float(precision_score(yt, yp, average="macro", labels=range(nc), zero_division=0)),
                rec=float(recall_score(yt, yp, average="macro", labels=range(nc), zero_division=0)),
                auc=auc_ovr(yt, P), ll=float(log_loss(yt, np.clip(P, 1e-7, 1), labels=range(nc))))


C = [collect(r) for r in reps]
models = list(C[0].keys())
NR = len(C)
# repeated predictions stacked: (NR, N, nc)
PP = {m: np.stack([c[m][0] for c in C]) for m in models}
boot_idx = [rng.randint(0, N, N) for _ in range(B)]


def boot(m_fn, m, key):
    """cluster bootstrap over samples (all repeats kept together)."""
    vals = []
    for idx in boot_idx:
        yt = np.tile(y[idx], NR); P = PP[m][:, idx].reshape(-1, nc)
        vals.append(m_fn(yt, P)[key] if key else m_fn(yt, P))
    return np.percentile(vals, [2.5, 97.5]).tolist()


summary = dict(ds=ds, N=int(N), nc=nc, classes=classes, n_reps=NR, models={})
fast = lambda yt, P: dict(acc=float((P.argmax(1) == yt).mean()), auc=auc_ovr(yt, P),
                          f1=float(f1_score(yt, P.argmax(1), average="macro", labels=range(nc), zero_division=0)))
for m in models:
    fold_metrics = [metrics(y[te], p) for c in C for te, p in c[m][1]]
    pooled = metrics(np.tile(y, NR), PP[m].reshape(-1, nc))
    bvals = [fast(np.tile(y[idx], NR), PP[m][:, idx].reshape(-1, nc)) for idx in boot_idx]
    ci = {k: np.percentile([b[k] for b in bvals], [2.5, 97.5]).tolist() for k in ["acc", "auc", "f1"]}
    summary["models"][m] = dict(
        pooled=pooled, ci=ci,
        fold_mean={k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]},
        fold_sd={k: float(np.std([f[k] for f in fold_metrics], ddof=1)) for k in fold_metrics[0]},
        fold_acc=[f["acc"] for f in fold_metrics], fold_auc=[f["auc"] for f in fold_metrics],
        fold_ll=[f["ll"] for f in fold_metrics], fold_f1=[f["f1"] for f in fold_metrics])
    print(f"{m:38s} acc={pooled['acc']:.4f} [{ci['acc'][0]:.3f},{ci['acc'][1]:.3f}] auc={pooled['auc']:.4f} [{ci['auc'][0]:.3f},{ci['auc'][1]:.3f}] f1={pooled['f1']:.4f} ll={pooled['ll']:.3f}")


def paired(a, b, key="acc"):
    """a vs b: corrected resampled t (Nadeau & Bengio 2003), Wilcoxon over folds,
    exact McNemar per repeat, cluster-bootstrap CI of the difference."""
    fa = np.array(summary["models"][a][f"fold_{key}"]); fb = np.array(summary["models"][b][f"fold_{key}"])
    d = fa - fb; k = len(d); ratio = 1 / 4  # n_test / n_train for 5-fold
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + ratio) * var) if var > 0 else 0.0
    p_t = float(2 * stats.t.sf(abs(t), k - 1)) if var > 0 else 1.0
    try:
        p_w = float(stats.wilcoxon(fa, fb, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = 1.0
    mc = []
    for r in range(NR):
        ca = PP[a][r].argmax(1) == y; cb = PP[b][r].argmax(1) == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        mc.append(dict(a_only=n01, b_only=n10, p=float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0))
    fn = (lambda yt, P: float((P.argmax(1) == yt).mean())) if key == "acc" else auc_ovr
    diffs = []
    for idx in boot_idx:
        yt = np.tile(y[idx], NR)
        diffs.append(fn(yt, PP[a][:, idx].reshape(-1, nc)) - fn(yt, PP[b][:, idx].reshape(-1, nc)))
    pooled_d = summary["models"][a]["pooled"][key] - summary["models"][b]["pooled"][key]
    disc = np.mean([(m_["a_only"] + m_["b_only"]) / N for m_ in mc])
    return dict(delta=float(pooled_d), fold_delta_mean=float(d.mean()), t=float(t), p_corrected_t=p_t, p_wilcoxon=p_w,
                mcnemar=mc, ci=np.percentile(diffs, [2.5, 97.5]).tolist(), discordant_rate=float(disc))


PROP = "CorrConvNet-Stack"
comparators = [m for m in ["LR", "SVM", "RF", "kNN", "MLP", "LGBM", "Vote", "DeepInsight", "CNN_corr", "Stack w/o CNN (classical only)"] if m in models]
summary["tests_vs_proposed"] = {}
for key in ["acc", "auc"]:
    res = {b: paired(PROP, b, key) for b in comparators}
    # Holm correction on corrected-t p values
    ps = np.array([res[b]["p_corrected_t"] for b in comparators]); order = np.argsort(ps); holm = np.empty_like(ps); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(ps) - i) * ps[j])); holm[j] = run
    for b, h in zip(comparators, holm):
        res[b]["p_holm"] = float(h)
    summary["tests_vs_proposed"][key] = res
# reordering ablation tests (stand-alone CNN and inside the stack)
summary["reorder_tests"] = {}
for key in ["acc", "auc"]:
    summary["reorder_tests"][key] = {
        "corr_vs_anova": paired("CNN_corr", "CNN_anova", key), "corr_vs_random": paired("CNN_corr", "CNN_random", key),
        "single_corr_vs_anova": paired("CNNs0_corr", "CNNs0_anova", key),
        "stack_corr_vs_stack_anova": paired(PROP, "Stack, CNN on ANOVA order", key)}
for o in ["corr", "anova", "random"]:
    fl = [np.mean([f["locality"][o] for f in r["folds"]]) for r in reps]
    summary.setdefault("locality", {})[o] = float(np.mean(fl))
summary["leakage"] = paired("Stack w/o CNN (classical only)", "Classical stack (leaky design)", "acc")
summary["leakage"]["leaky_oof_acc"] = float(np.mean([np.mean(f["oof"]["LR_leaky"].argmax(1) == f["y_tr"]) for r in reps for f in r["folds"]]))
summary["leakage"]["nested_oof_acc"] = float(np.mean([np.mean(f["oof"]["LR"].argmax(1) == f["y_tr"]) for r in reps for f in r["folds"]]))

# power (paired / McNemar, Connor 1987) for the proposed-vs-best-classical gap
best = max(["LR", "SVM", "RF"], key=lambda m: summary["models"][m]["pooled"]["acc"])
tv = summary["tests_vs_proposed"]["acc"][best]; dlt = abs(tv["delta"]); pd_ = max(tv["discordant_rate"], dlt + 1e-6)
za, zb = 1.959964, 0.841621
n_req = ((za * np.sqrt(pd_) + zb * np.sqrt(max(pd_ - dlt ** 2, 1e-12))) ** 2) / max(dlt ** 2, 1e-12) if dlt > 0 else float("inf")
mde = None
for dd in np.arange(0.001, 0.5, 0.001):
    if ((za * np.sqrt(pd_) + zb * np.sqrt(max(pd_ - dd ** 2, 1e-12))) ** 2) / dd ** 2 <= N:
        mde = float(dd); break
summary["power"] = dict(best_classical=best, delta=dlt, discordant=pd_, n_required=float(n_req), mde_at_N=mde)

# permutation sanity check
if perm:
    pc = collect(perm[0])
    summary["permutation"] = {m: float((pc[m][0].argmax(1) == perm[0]["y"]).mean()) for m in pc}
    summary["permutation"]["chance_majority"] = float(np.bincount(perm[0]["y"]).max() / N)

# per-class metrics for proposed (pooled over repeats) with bootstrap CIs
yt_all = np.tile(y, NR); P_all = PP[PROP].reshape(-1, nc); yp_all = P_all.argmax(1)
pcm = []
for c in range(nc):
    def pcfn(yt, P, c=c):
        yp = P.argmax(1); tp = ((yp == c) & (yt == c)).sum(); fp = ((yp == c) & (yt != c)).sum(); fn_ = ((yp != c) & (yt == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn_) if tp + fn_ else 0
        pos = yt == c; r = rankdata(P[:, c]); npos, nneg = pos.sum(), (~pos).sum()
        au = (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else np.nan
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0), au
    pt = pcfn(yt_all, P_all)
    bs = np.array([pcfn(np.tile(y[idx], NR), PP[PROP][:, idx].reshape(-1, nc)) for idx in boot_idx[:1000]])
    pcm.append(dict(cls=classes[c], n=int((y == c).sum()), prec=float(pt[0]), rec=float(pt[1]), f1=float(pt[2]), auc=float(pt[3]),
                    f1_ci=np.nanpercentile(bs[:, 2], [2.5, 97.5]).tolist(), auc_ci=np.nanpercentile(bs[:, 3], [2.5, 97.5]).tolist()))
summary["per_class"] = pcm
summary["confusion"] = confusion_matrix(yt_all, yp_all, labels=range(nc)).tolist()

# XAI faithfulness (repeat 0)
r0 = [r for r in reps if r["rep"] == 0][0]
if "xai" in r0["folds"][0]:
    from xai import KS
    agg = {}
    for nm in r0["folds"][0]["xai"]["curves"]:
        p0 = np.concatenate([f["xai"]["curves"][nm][0] for f in r0["folds"]])
        de = np.concatenate([f["xai"]["curves"][nm][1] for f in r0["folds"]])
        ins = np.concatenate([f["xai"]["curves"][nm][2] for f in r0["folds"]])
        agg[nm] = dict(p0=p0, dele=de, ins=ins)
    xs = {}
    for nm, v in agg.items():
        aopc = (v["p0"][:, None] - v["dele"]).mean(1)
        xs[nm] = dict(aopc=float(aopc.mean()), aopc_sd=float(aopc.std(ddof=1)),
                      del_curve=v["dele"].mean(0).tolist(), ins_curve=v["ins"].mean(0).tolist(),
                      ins_mean=float(v["ins"].mean()), p0=float(v["p0"].mean()), aopc_per=aopc.tolist())
    for nm in list(xs):
        ref = "Random" if "_" not in nm else "Random_" + nm.split("_")[1]
        if nm != ref and ref in xs:
            xs[nm]["p_vs_random"] = float(stats.wilcoxon(xs[nm]["aopc_per"], xs[ref]["aopc_per"]).pvalue)
    for o in ["anova", "random"]:
        a_, b_ = np.array(xs["GradCAM++"]["aopc_per"]), np.array(xs["GradCAM++_" + o]["aopc_per"])
        xs["GradCAM++"][f"p_vs_order_{o}"] = float(stats.wilcoxon(a_, b_).pvalue)
        xs["GradCAM++"][f"gain_vs_random_ratio"] = float((a_.mean() - np.mean(xs["Random"]["aopc_per"])))
    for nm in xs:
        xs[nm].pop("aopc_per")
    # stability & agreement
    sp, jac, sp_gl = [], [], []
    for f in r0["folds"]:
        fx = f["xai"]; G = fx["gradcam_seeds"]
        for i in range(G.shape[1]):
            sp.append(np.mean([stats.spearmanr(G[a, i], G[b, i])[0] for a, b in [(0, 1), (0, 2), (1, 2)]]))
            tg = set(np.argsort(-fx["gradcam"][i])[:20]); tl = set(np.argsort(-np.abs(fx["lime"][i]))[:20])
            jac.append(len(tg & tl) / len(tg | tl)); sp_gl.append(stats.spearmanr(fx["gradcam"][i], np.abs(fx["lime"][i]))[0])
    # LIME stability: not re-run per seed (cost) -> report Grad-CAM++ seed stability only
    summary["xai"] = dict(ks=KS, methods=xs, gradcam_seed_spearman=float(np.nanmean(sp)),
                          gc_lime_top20_jaccard=float(np.mean(jac)), gc_lime_spearman=float(np.nanmean(sp_gl)))
    pickle.dump(dict(agg=agg, fold0=r0["folds"][0]["xai"], te0=r0["folds"][0]["te"], curve=r0.get("curve")), open(f"{R}/xai_raw.pkl", "wb"))
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ["locality", "power"]}, indent=1))
print("perm", summary.get("permutation"))
print("leak", {k: v for k, v in summary["leakage"].items() if k != "mcnemar"})
