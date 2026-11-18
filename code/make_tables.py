"""Generate LaTeX tables + number macros directly from summary.json (no hand transcription)."""
import json
S = {d: json.load(open(f"/home/claude/results/{d}/summary.json")) for d in ["11tumors", "leukemia"]}
T, L = S["11tumors"], S["leukemia"]
P = "CorrConvNet-Stack"
f3 = lambda v: f"{v:.3f}"
pc = lambda v: f"{100*v:.1f}"
ci = lambda c: f"[{100*c[0]:.1f}, {100*c[1]:.1f}]"
ci3 = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"
out = []
LAB = {"CorrConvNet-Stack": "\\textbf{CorrConvNet-Stack (ours)}", "LR": "LogReg-L2 (CV-tuned)", "SVM": "SVM-RBF", "RF": "Random Forest",
       "kNN": "$k$NN ($k{=}3$)", "MLP": "MLP (256-64)", "LGBM": "LightGBM \\cite{lightgbm}", "Vote": "Voting SVM+$k$NN+DT \\cite{khatun2023}",
       "DeepInsight": "DeepInsight-CNN \\cite{deepinsight}", "CNN_corr": "CorrConvNet-1D alone"}

# ---- Table: main comparison
rows = []
for m in LAB:
    a, b = T["models"][m], L["models"][m]
    if m == P:
        d = p = "--"
    else:
        t = T["tests_vs_proposed"]["acc"][m]
        d = f"{100*t['delta']:+.1f} {ci(t['ci'])}"; p = f"{pv(t['p_corrected_t'])} / {pv(t['p_holm'])}"
    rows.append(f"{LAB[m]} & {pc(a['pooled']['acc'])} {ci(a['ci']['acc'])} & {f3(a['pooled']['f1'])} & {f3(a['pooled']['auc'])} & {d} & {p} & {pc(b['pooled']['acc'])} & {f3(b['pooled']['auc'])}\\\\")
out.append(r"""\begin{table}[t]\centering
\caption{Repeated stratified 5-fold CV ($3\times5$ folds, pooled out-of-fold predictions). Acc.\ in \%, 95\% cluster-bootstrap CI in brackets. $\Delta$ = accuracy of CorrConvNet-Stack minus that of the row model, with paired bootstrap CI. $p$ = variance-corrected resampled $t$-test (raw / Holm-adjusted over the 9 comparisons).}\label{tab:main}
\scriptsize\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{l ccc cc cc}\toprule
 & \multicolumn{5}{c}{11\_Tumors ($N{=}174$, 11 classes)} & \multicolumn{2}{c}{Leukemia ($N{=}64$, 5 cl.)}\\\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Model & Acc. [95\% CI] & F1$_\mathrm{macro}$ & AUC & $\Delta$Acc [95\% CI] & $p$ (raw/Holm) & Acc. & AUC\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}")

# ---- Table: reordering ablation
rr = []
nm = {"corr": "(A) Correlation reordering", "anova": "(B) ANOVA-rank order", "random": "(C) Random permutation"}
for o in ["corr", "anova", "random"]:
    a, b = T["models"][f"CNN_{o}"], L["models"][f"CNN_{o}"]
    rr.append(f"{nm[o]} & {T['locality'][o]:.2f} & {pc(a['pooled']['acc'])} {ci(a['ci']['acc'])} & {f3(a['pooled']['auc'])} & {a['pooled']['ll']:.3f} & {L['locality'][o]:.2f} & {pc(b['pooled']['acc'])} {ci(b['ci']['acc'])} & {f3(b['pooled']['auc'])} & {b['pooled']['ll']:.3f}\\\\")
def tl(S_, k):
    t = S_["reorder_tests"]["acc"][k]
    return f"{100*t['delta']:+.1f} {ci(t['ci'])}, $p_t$={pv(t['p_corrected_t'])}, $p_W$={pv(t['p_wilcoxon'])}"
rr.append("\\midrule A$-$B ($\\Delta$Acc, CI, tests) & \\multicolumn{4}{c}{" + tl(T, "corr_vs_anova") + "} & \\multicolumn{4}{c}{" + tl(L, "corr_vs_anova") + "}\\\\")
rr.append("A$-$C & \\multicolumn{4}{c}{" + tl(T, "corr_vs_random") + "} & \\multicolumn{4}{c}{" + tl(L, "corr_vs_random") + "}\\\\")
rr.append("Inside the stack: A$-$B & \\multicolumn{4}{c}{" + tl(T, "stack_corr_vs_stack_anova") + "} & \\multicolumn{4}{c}{" + tl(L, "stack_corr_vs_stack_anova") + "}\\\\")
out.append(r"""\begin{table}[t]\centering
\caption{Full-architecture reordering ablation: the complete CorrConvNet-1D (3-seed ensemble) trained on identical folds/seeds under three input orders. Loc.\ = mean $|r|$ between adjacent genes (training folds). LL = log-loss (lower is better). $p_t$: corrected resampled $t$-test; $p_W$: Wilcoxon over the 15 folds.}\label{tab:reorder}
\scriptsize\setlength{\tabcolsep}{2pt}
\begin{tabular}{l cccc cccc}\toprule
 & \multicolumn{4}{c}{11\_Tumors} & \multicolumn{4}{c}{Leukemia}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Input order & Loc. & Acc. [95\% CI] & AUC & LL & Loc. & Acc. [95\% CI] & AUC & LL\\\midrule
""" + "\n".join(rr) + "\n\\bottomrule\\end{tabular}\\end{table}")

# ---- Table: stack ablation
SA = [("CorrConvNet-Stack (LR+SVM+RF+CNN, LR meta)", P), ("CNN only (3-seed CorrConvNet-1D)", "CNN_corr"),
      ("Classical only (stack w/o CNN)", "Stack w/o CNN (classical only)"), ("Stack w/o LogReg", "Stack w/o LR"),
      ("Stack w/o SVM", "Stack w/o SVM"), ("Stack w/o RF", "Stack w/o RF"), ("Stack, CNN on ANOVA order", "Stack, CNN on ANOVA order"),
      ("Meta = probability average", "Stack, meta = average"), ("Meta = LR, $C{=}0.1$", "Stack, meta = LR (C=0.1)"),
      ("Meta = Random Forest", "Stack, meta = RF"), ("Meta = Gaussian NB", "Stack, meta = Gaussian NB")]
sr = []
for lab, m in SA:
    a, b = T["models"][m], L["models"][m]
    sr.append(f"{lab} & {pc(a['pooled']['acc'])} {ci(a['ci']['acc'])} & {f3(a['pooled']['f1'])} & {f3(a['pooled']['auc'])} & {a['pooled']['ll']:.3f} & {pc(b['pooled']['acc'])} & {f3(b['pooled']['auc'])}\\\\")
    if m == "CNN_corr" or m == "Stack, CNN on ANOVA order":
        sr.append("\\midrule")
out.append(r"""\begin{table}[t]\centering
\caption{Stack-component ablation (same folds as Table~\ref{tab:main}). All variants are fully nested.}\label{tab:stack}
\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l cccc cc}\toprule
 & \multicolumn{4}{c}{11\_Tumors} & \multicolumn{2}{c}{Leukemia}\\\cmidrule(lr){2-5}\cmidrule(lr){6-7}
Variant & Acc. [95\% CI] & F1$_\mathrm{macro}$ & AUC & LL & Acc. & AUC\\\midrule
""" + "\n".join(sr) + "\n\\bottomrule\\end{tabular}\\end{table}")

# ---- Table: XAI faithfulness
xr = []
for d, S_ in [("11\\_Tumors", T), ("Leukemia", L)]:
    xm = S_["xai"]["methods"]
    for lab, k in [("Grad-CAM++ (A: correlation order)", "GradCAM++"), ("Grad-CAM++ (B: ANOVA order)", "GradCAM++_anova"),
                   ("Grad-CAM++ (C: random order)", "GradCAM++_random"), ("LIME (A)", "LIME"), ("Gradient$\\times$Input (A)", "GradxInput"), ("Random ranking (A)", "Random")]:
        v = xm[k]
        p = pv(v["p_vs_random"]) if "p_vs_random" in v else "--"
        xr.append(f"{d} & {lab} & {v['aopc']:.3f} & {v['aopc']/v['p0']:.3f} & {v['ins_mean']/v['p0']:.3f} & {p}\\\\")
        d = ""
    xr.append(f"& \\multicolumn{{5}}{{l}}{{Grad-CAM++ A vs.\\ B: $p$={pv(xm['GradCAM++']['p_vs_order_anova'])}; A vs.\\ C: $p$={pv(xm['GradCAM++']['p_vs_order_random'])}; seed stability $\\rho$={S_['xai']['gradcam_seed_spearman']:.2f}}}\\\\")
    xr.append(f"& \\multicolumn{{5}}{{l}}{{Grad-CAM++/LIME top-20 Jaccard = {S_['xai']['gc_lime_top20_jaccard']:.3f} (chance $\\approx$0.013)}}\\\\\\midrule")
xr[-1] = xr[-1].replace("\\midrule", "")
out.append(r"""\begin{table}[t]\centering
\caption{Quantitative XAI faithfulness on all held-out samples of repeat 1 (deletion/insertion of the top-$k$ attributed genes, $k\in\{1,2,5,\dots,400\}$). AOPC = mean drop of the predicted-class probability (higher = more faithful); nAOPC and nIns are normalised by the unperturbed probability $p_0$. $p$: Wilcoxon signed-rank vs.\ the random ranking of the same model. Random baselines have AOPC 0.03--0.05.}\label{tab:xai}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll cccc}\toprule
Data & Explainer (model input order) & AOPC$\uparrow$ & nAOPC$\uparrow$ & nIns$\uparrow$ & $p$ vs.\ random\\\midrule
""" + "\n".join(xr) + "\n\\bottomrule\\end{tabular}\\end{table}")

# ---- Table: per-class (11T)
pr = []
for c in T["per_class"]:
    pr.append(f"{c['cls']} & {c['n']} & {c['prec']:.2f} & {c['rec']:.2f} & {c['f1']:.2f} {ci3(c['f1_ci']).replace('0.', '.')} & {c['auc']:.3f} {ci3(c['auc_ci']).replace('0.', '.')}\\\\")
out.append(r"""\begin{table}[t]\centering
\caption{Per-class metrics of CorrConvNet-Stack on 11\_Tumors (pooled over the 3 repeats; 95\% bootstrap CIs). Class indices follow the public CSV.}\label{tab:perclass}
\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{cc cccc}\toprule
Class & $n$ & Prec. & Rec. & F1 [95\% CI] & AUC [95\% CI]\\\midrule
""" + "\n".join(pr) + "\n\\bottomrule\\end{tabular}\\end{table}")
open("/home/claude/paper/tables.tex", "w").write("\n\n".join(out))

# ---- macros
M = {}
def put(k, v): M[k] = v
p_ = T["models"][P]
put("PAcc", pc(p_["pooled"]["acc"])); put("PAccCI", ci(p_["ci"]["acc"])); put("PAUC", f3(p_["pooled"]["auc"])); put("PAUCCI", ci3(p_["ci"]["auc"]))
put("PFone", f3(p_["pooled"]["f1"])); put("PFoldSD", pc(p_["fold_sd"]["acc"]))
for m, k in [("LR", "LR"), ("LGBM", "LGBM"), ("SVM", "SVM"), ("RF", "RF"), ("CNN_corr", "CNN")]:
    put(f"{k.replace('_corr','')}Acc", pc(T["models"][m]["pooled"]["acc"])); put(f"{k.replace('_corr','')}AUC", f3(T["models"][m]["pooled"]["auc"]))
t = T["tests_vs_proposed"]["acc"]["LR"]
put("DLR", f"{100*t['delta']:+.1f}"); put("DLRCI", ci(t["ci"])); put("pLR", pv(t["p_corrected_t"])); put("pLRW", pv(t["p_wilcoxon"]))
put("McLR", ", ".join(f"{m_['a_only']}/{m_['b_only']}" for m_ in t["mcnemar"]))
ta = T["tests_vs_proposed"]["auc"]["LR"]; put("DAUCLR", f"{ta['delta']:+.4f}"); put("DAUCLRCI", ci3(ta["ci"]))
put("NReq", f"{T['power']['n_required']:,.0f}".replace(",", "{,}")); put("MDE", pc(T["power"]["mde_at_N"])); put("Disc", pc(T["power"]["discordant"]))
lk = T["leakage"]; put("LeakOOF", pc(lk["leaky_oof_acc"])); put("NestOOF", pc(lk["nested_oof_acc"]))
put("LeakTest", pc(T["models"]["Classical stack (leaky design)"]["pooled"]["acc"])); put("NestTest", pc(T["models"]["Stack w/o CNN (classical only)"]["pooled"]["acc"]))
for d, S_ in [("T", T), ("L", L)]:
    if "permutation" in S_:
        put(f"Perm{d}", pc(S_["permutation"]["CorrConvNet-Stack"])); put(f"PermMaj{d}", pc(S_["permutation"]["chance_majority"]))
    for o in ["corr", "anova", "random"]:
        put(f"Loc{d}{o}", f"{S_['locality'][o]:.2f}"); put(f"CNN{d}{o}", pc(S_["models"][f"CNN_{o}"]["pooled"]["acc"]))
    ra = S_["reorder_tests"]["acc"]
    put(f"RAB{d}", f"{100*ra['corr_vs_anova']['delta']:+.1f}"); put(f"RABCI{d}", ci(ra["corr_vs_anova"]["ci"])); put(f"RABp{d}", pv(ra["corr_vs_anova"]["p_corrected_t"]))
    put(f"RAC{d}", f"{100*ra['corr_vs_random']['delta']:+.1f}"); put(f"RACCI{d}", ci(ra["corr_vs_random"]["ci"])); put(f"RACp{d}", pv(ra["corr_vs_random"]["p_corrected_t"]))
    xm = S_["xai"]["methods"]
    put(f"GC{d}", f"{xm['GradCAM++']['aopc']:.3f}"); put(f"GCB{d}", f"{xm['GradCAM++_anova']['aopc']:.3f}"); put(f"GCC{d}", f"{xm['GradCAM++_random']['aopc']:.3f}")
    put(f"GCpB{d}", pv(xm["GradCAM++"]["p_vs_order_anova"])); put(f"GCpC{d}", pv(xm["GradCAM++"]["p_vs_order_random"]))
    put(f"Jac{d}", f"{S_['xai']['gc_lime_top20_jaccard']:.3f}"); put(f"Stab{d}", f"{S_['xai']['gradcam_seed_spearman']:.2f}")
    put(f"PAcc{d}", pc(S_["models"][P]["pooled"]["acc"])); put(f"LRAcc{d}", pc(S_["models"]["LR"]["pooled"]["acc"]))
mac = "\n".join(f"\\newcommand{{\\{k.replace('1','One').replace('0','Zero')}}}{{{v}}}" for k, v in M.items())
open("/home/claude/paper/numbers.tex", "w").write(mac + "\n")
print(mac)
