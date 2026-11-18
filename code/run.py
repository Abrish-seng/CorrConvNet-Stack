"""One (dataset, repeat) job of the revision experiments. Saves everything needed for
analysis to results/<ds>/rep<r>.pkl.

python run.py --ds 11tumors --rep 0            # main job
python run.py --ds 11tumors --rep 0 --perm     # label-permutation leakage sanity check
"""
import argparse, os, pickle, time, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data import load
from corrconv import Prep, CNNEnsemble, Wrapped, DeepInsight, inner_oof, train_cnn, predict_cnn
import xai

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--rep", type=int, default=0)
ap.add_argument("--perm", action="store_true")
ap.add_argument("--k", type=int, default=800)
a = ap.parse_args()

X, y, names, classes = load(a.ds)
nc = len(classes)
if a.perm:
    y = np.random.RandomState(999).permutation(y)
outdir = os.environ.get("CC_OUT", "/home/claude/results") + f"/{a.ds}"
os.makedirs(outdir, exist_ok=True)
fn = f"{outdir}/{'perm' if a.perm else 'rep'}{a.rep}.pkl"
skf = StratifiedKFold(5, shuffle=True, random_state=100 + a.rep)
log = lambda *s: print(f"[{a.ds} r{a.rep}{' perm' if a.perm else ''} {time.strftime('%H:%M:%S')}]", *s, flush=True)

res = dict(ds=a.ds, rep=a.rep, perm=a.perm, y=y, classes=classes, folds=[])
for f, (tr, te) in enumerate(skf.split(X, y)):
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    fold = dict(tr=tr, te=te, test={}, oof={}, locality={}, cols={})
    inner_seed = 1000 * a.rep + f

    # --- base learners used inside the stack: inner OOF (nested) + outer test ---
    stack_bases = {
        "LR": lambda: Wrapped("LR", nc, a.k),
        "SVM": lambda: Wrapped("SVM", nc, a.k),
        "RF": lambda: Wrapped("RF", nc, a.k),
        "CNNs0_corr": lambda: CNNEnsemble(nc, "corr", a.k, seeds=(0,)),
    }
    if not a.perm:
        stack_bases["CNNs0_anova"] = lambda: CNNEnsemble(nc, "anova", a.k, seeds=(0,))
    for nm, fac in stack_bases.items():
        fold["oof"][nm] = inner_oof(fac, Xtr, ytr, nc, 3, seed=inner_seed)
        log(f"fold {f} inner OOF {nm} done")
    # leaky OOF variant for classical learners (Prep fitted ONCE on the whole outer-train,
    # then inner CV) -> quantifies the optimism of the leaky stacking design
    P = Prep(a.k, "anova").fit(Xtr, ytr); Ztr = P.transform(Xtr)
    from corrconv import make_classical
    for nm in ["LR", "SVM", "RF"]:
        oof = np.zeros((len(ytr), nc))
        for itr, iva in StratifiedKFold(3, shuffle=True, random_state=inner_seed).split(Ztr, ytr):
            c = make_classical(nm).fit(Ztr[itr], ytr[itr])
            pp = np.zeros((len(iva), nc)); pp[:, c.classes_] = c.predict_proba(Ztr[iva]); oof[iva] = pp
        fold["oof"][nm + "_leaky"] = oof

    # --- outer models ---
    cnn = {}
    orders = ["corr"] if a.perm else ["corr", "anova", "random"]
    for o in orders:
        seeds = (0,) if a.perm else (0, 1, 2)
        e = CNNEnsemble(nc, o, a.k, seeds=seeds, prep_seed=inner_seed).fit(Xtr, ytr)
        cnn[o] = e
        fold["test"][f"CNN_{o}"] = e.predict_proba(Xte)
        Zs = e.prep.transform(Xte)
        fold["test"][f"CNNs0_{o}"] = predict_cnn(e.models[:1], Zs)
        Zt = e.prep.transform(Xtr)
        C = np.corrcoef(Zt.T)
        fold["locality"][o] = float(np.nanmean(np.abs(np.diag(C, 1))))
        fold["cols"][o] = e.prep.cols
        log(f"fold {f} CNN {o} acc={np.mean(fold['test'][f'CNN_{o}'].argmax(1)==y[te]):.3f}")
    base_names = ["LR", "SVM", "RF"] if a.perm else ["LR", "SVM", "RF", "kNN", "MLP", "LGBM", "Vote"]
    for nm in base_names:
        fold["test"][nm] = Wrapped(nm, nc, a.k).fit(Xtr, ytr).predict_proba(Xte)
    if not a.perm:
        fold["test"]["DeepInsight"] = DeepInsight(nc, a.k).fit(Xtr, ytr).predict_proba(Xte)
    log(f"fold {f} baselines done " + " ".join(f"{k}={np.mean(v.argmax(1)==y[te]):.3f}" for k, v in fold["test"].items()))

    # --- XAI (repeat 0 only) ---
    if a.rep == 0 and not a.perm:
        e = cnn["corr"]; Zte = e.prep.transform(Xte); Ztr_c = e.prep.transform(Xtr)
        cls = fold["test"]["CNN_corr"].argmax(1)
        gc, gc_seeds = xai.gradcampp(e.models, Zte, cls)
        gxi = xai.grad_x_input(e.models, Zte, cls)
        lw = xai.lime_weights(e.models, Ztr_c, Zte, cls, seed=inner_seed)
        rnd = np.random.RandomState(inner_seed).rand(*Zte.shape)
        fx = dict(cls=cls, gradcam=gc, gradcam_seeds=gc_seeds, gxi=gxi, lime=lw, cols=e.prep.cols, Zte=Zte, curves={})
        for nm, sc in [("GradCAM++", gc), ("GradxInput", gxi), ("LIME", np.abs(lw)), ("Random", rnd)]:
            fx["curves"][nm] = xai.deletion_insertion(e.models, Zte, cls, sc)
        # Grad-CAM++ faithfulness as a function of input ordering (locality assumption)
        for o in ["anova", "random"]:
            eo = cnn[o]; Zo = eo.prep.transform(Xte); co = fold["test"][f"CNN_{o}"].argmax(1)
            g_o, _ = xai.gradcampp(eo.models, Zo, co)
            r_o = np.random.RandomState(inner_seed + 7).rand(*Zo.shape)
            fx["curves"][f"GradCAM++_{o}"] = xai.deletion_insertion(eo.models, Zo, co, g_o)
            fx["curves"][f"Random_{o}"] = xai.deletion_insertion(eo.models, Zo, co, r_o)
        fold["xai"] = fx
        log(f"fold {f} XAI done")
        if f == 0:
            Ztr0 = cnn["corr"].prep.transform(Xtr)
            _, curve = train_cnn(Ztr0, ytr, nc, 0, return_curve=True, Xva=Zte, yva=y[te])
            res["curve"] = curve
    res["folds"].append(fold)
    pickle.dump(res, open(fn, "wb"))
log("DONE")
