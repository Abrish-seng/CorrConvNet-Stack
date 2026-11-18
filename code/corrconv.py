"""CorrConvNet-Stack: fully nested, leakage-free implementation used for the revision.

Every data-dependent preprocessing step (zero-variance filter, ANOVA-F selection,
z-scoring, correlation-based reordering) is re-fitted on whichever training split
is being used: the outer training fold for test predictions, and each inner
training split for the out-of-fold (OOF) meta-features that train the stacker.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

torch.set_num_threads(1)


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
class Prep:
    """Zero-variance filter -> ANOVA top-k -> z-score -> column ordering.

    order: 'corr'   = average-linkage hierarchical clustering on 1-|Pearson r| (proposed)
           'anova'  = descending ANOVA-F rank (arbitrary w.r.t. co-expression)
           'random' = uniform random permutation (seeded)
    """

    def __init__(self, k=800, order="corr", seed=0):
        self.k, self.order, self.seed = k, order, seed

    def fit(self, X, y):
        var = X.var(0)
        nz = np.where(var > 1e-12)[0]
        F_, _ = f_classif(X[:, nz], y)
        F_ = np.nan_to_num(F_, nan=0.0)
        top = nz[np.argsort(-F_)[: self.k]]  # descending F
        Xs = X[:, top]
        self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-8
        Z = (Xs - self.mu) / self.sd
        if self.order == "corr":
            C = np.corrcoef(Z.T)
            C = np.nan_to_num(C)
            D = 1.0 - np.abs(C)
            np.fill_diagonal(D, 0.0)
            D = np.clip((D + D.T) / 2, 0, None)
            L = linkage(squareform(D, checks=False), method="average")
            perm = leaves_list(L)
        elif self.order == "anova":
            perm = np.arange(len(top))
        elif self.order == "random":
            perm = np.random.RandomState(self.seed).permutation(len(top))
        else:
            raise ValueError(self.order)
        self.cols = top[perm]
        self.mu, self.sd = self.mu[perm], self.sd[perm]
        return self

    def transform(self, X):
        return ((X[:, self.cols] - self.mu) / self.sd).astype(np.float32)


# ----------------------------------------------------------------------------
# CorrConvNet-1D
# ----------------------------------------------------------------------------
class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.f1, self.f2 = nn.Linear(c, max(c // r, 4)), nn.Linear(max(c // r, 4), c)

    def forward(self, x):
        s = x.mean(-1)
        s = torch.sigmoid(self.f2(F.gelu(self.f1(s))))
        return x * s.unsqueeze(-1)


class CorrConvBlock(nn.Module):
    def __init__(self, c, k=7, p=0.3):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.norm = nn.GroupNorm(1, c)
        self.pw1, self.pw2 = nn.Conv1d(c, 4 * c, 1), nn.Conv1d(4 * c, c, 1)
        self.se, self.drop = SE(c), nn.Dropout(p)

    def forward(self, x):
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return x + self.drop(self.se(h))


class CorrConvNet1D(nn.Module):
    def __init__(self, n_classes, widths=(24, 48, 96), depth=2, k=7, p=0.3):
        super().__init__()
        self.stem = nn.Conv1d(1, widths[0], 9, stride=2, padding=4)
        stages, downs = [], []
        for i, w in enumerate(widths):
            stages.append(nn.Sequential(*[CorrConvBlock(w, k, p) for _ in range(depth)]))
            if i < len(widths) - 1:
                downs.append(nn.Sequential(nn.GroupNorm(1, w), nn.Conv1d(w, widths[i + 1], 2, stride=2)))
        self.stages, self.downs = nn.ModuleList(stages), nn.ModuleList(downs)
        self.norm = nn.GroupNorm(1, widths[-1])
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(widths[-1], n_classes)

    def features(self, x):
        x = self.stem(x.unsqueeze(1))
        for i, s in enumerate(self.stages):
            x = s(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return x  # (B, C, T) final conv feature map -> Grad-CAM++ target

    def forward_from_features(self, A):
        return self.head(self.drop(self.norm(A).mean(-1)))

    def forward(self, x):
        return self.forward_from_features(self.features(x))


def train_cnn(Xtr, ytr, n_classes, seed, epochs=None, bs=16, lr=1e-3, wd=5e-3,
              mix_alpha=0.3, noise=0.05, return_curve=False, Xva=None, yva=None):
    epochs = epochs or int(os.environ.get("CC_EPOCHS", 70))
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    m = CorrConvNet1D(n_classes)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps_per_ep = int(np.ceil(len(Xtr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep)
    cnt = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(len(ytr) / (n_classes * np.maximum(cnt, 1)), dtype=torch.float32)
    X = torch.tensor(Xtr)
    Y = F.one_hot(torch.tensor(ytr), n_classes).float()
    curve = []
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(X))
        tot = 0.0
        for b in range(steps_per_ep):
            bi = idx[b * bs:(b + 1) * bs]
            xb, yb = X[bi], Y[bi]
            if mix_alpha > 0 and len(bi) > 1:
                lam = rng.beta(mix_alpha, mix_alpha)
                j = torch.tensor(rng.permutation(len(bi)))
                xb = lam * xb + (1 - lam) * xb[j]
                yb = lam * yb + (1 - lam) * yb[j]
            xb = xb + noise * torch.randn_like(xb)
            logp = F.log_softmax(m(xb), -1)
            loss = -(yb * logp * w).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(bi)
        if return_curve:
            va = float((predict_cnn([m], Xva).argmax(1) == yva).mean()) if Xva is not None else np.nan
            curve.append((tot / len(X), va))
    m.eval()
    return (m, curve) if return_curve else m


@torch.no_grad()
def predict_cnn(models, X):
    X = torch.tensor(X)
    ps = []
    for m in models:
        m.eval()
        ps.append(F.softmax(m(X), -1).numpy())
    return np.mean(ps, 0)


class CNNEnsemble:
    """Multi-seed CorrConvNet-1D ensemble with its own Prep (order configurable)."""

    def __init__(self, n_classes, order="corr", k=800, seeds=(0, 1, 2), epochs=None, prep_seed=0):
        self.nc, self.order, self.k, self.seeds, self.epochs, self.prep_seed = n_classes, order, k, seeds, epochs, prep_seed

    def fit(self, X, y):
        self.prep = Prep(self.k, self.order, self.prep_seed).fit(X, y)
        Z = self.prep.transform(X)
        self.models = [train_cnn(Z, y, self.nc, s, epochs=self.epochs) for s in self.seeds]
        return self

    def predict_proba(self, X):
        return predict_cnn(self.models, self.prep.transform(X))


# ----------------------------------------------------------------------------
# Classical learners (each wrapped with its own Prep so that it is nested)
# ----------------------------------------------------------------------------
def make_classical(name, seed=0):
    if name == "LR":
        return LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, max_iter=3000, n_jobs=1)
    if name == "SVM":
        return SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=3, weights="distance")
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, max_iter=1000,
                             early_stopping=False, random_state=seed)
    if name == "LGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=3, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.3, random_state=seed, verbose=-1, n_jobs=1)
    if name == "Vote":  # Khatun et al. (2023)-style soft voting SVM + kNN + DT
        return VotingClassifier([
            ("svm", SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=seed)),
            ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance")),
            ("dt", DecisionTreeClassifier(random_state=seed))], voting="soft")
    raise ValueError(name)


class Wrapped:
    def __init__(self, name, n_classes, k=800, seed=0):
        self.name, self.nc, self.k, self.seed = name, n_classes, k, seed

    def fit(self, X, y):
        self.prep = Prep(self.k, "anova").fit(X, y)  # order irrelevant for these learners
        self.clf = make_classical(self.name, self.seed).fit(self.prep.transform(X), y)
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), self.nc))
        p[:, self.clf.classes_] = self.clf.predict_proba(self.prep.transform(X))
        return p


# ----------------------------------------------------------------------------
# DeepInsight-style 2D CNN baseline (Sharma et al., 2019)
# ----------------------------------------------------------------------------
class SmallCNN2D(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv2d(1, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm2d(16), nn.BatchNorm2d(32), nn.BatchNorm2d(64)
        self.drop, self.fc = nn.Dropout(0.3), nn.Linear(64, nc)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.c2(x))), 2)
        x = F.relu(self.bn3(self.c3(x)))
        return self.fc(self.drop(x.mean((-1, -2))))


class DeepInsight:
    def __init__(self, n_classes, k=800, grid=32, seeds=(0, 1, 2), epochs=None):
        self.nc, self.k, self.g, self.seeds = n_classes, k, grid, seeds
        self.epochs = epochs or int(os.environ.get("CC_EPOCHS", 70))

    def _img(self, Z):
        n = len(Z)
        img = np.zeros((n, self.g * self.g), np.float32)
        np.add.at(img.T, self.pix, Z.T)
        img = img / np.maximum(self.cnt, 1)
        return img.reshape(n, 1, self.g, self.g)

    def fit(self, X, y):
        self.prep = Prep(self.k, "anova").fit(X, y)
        Z = self.prep.transform(X)
        emb = TSNE(2, perplexity=min(30, (Z.shape[1] - 1) / 3), init="pca", random_state=0,
                   metric="correlation").fit_transform(Z.T)
        # convex-hull-free min/max scaling onto the grid (DeepInsight Sec. Methods)
        e = (emb - emb.min(0)) / (np.ptp(emb, 0) + 1e-9)
        ij = np.clip((e * (self.g - 1)).round().astype(int), 0, self.g - 1)
        self.pix = ij[:, 0] * self.g + ij[:, 1]
        self.cnt = np.bincount(self.pix, minlength=self.g * self.g).astype(np.float32)
        I = torch.tensor(self._img(Z))
        Y = F.one_hot(torch.tensor(y), self.nc).float()
        self.models = []
        for s in self.seeds:
            torch.manual_seed(s); rng = np.random.RandomState(s)
            m = SmallCNN2D(self.nc)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-3)
            spe = int(np.ceil(len(I) / 16))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs * spe)
            for ep in range(self.epochs):
                m.train(); idx = rng.permutation(len(I))
                for b in range(spe):
                    bi = idx[b * 16:(b + 1) * 16]
                    if len(bi) < 2:
                        continue
                    loss = -(Y[bi] * F.log_softmax(m(I[bi] + 0.05 * torch.randn_like(I[bi])), -1)).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            m.eval(); self.models.append(m)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        I = torch.tensor(self._img(self.prep.transform(X)))
        return np.mean([F.softmax(m(I), -1).numpy() for m in self.models], 0)


# ----------------------------------------------------------------------------
# Stacking utilities
# ----------------------------------------------------------------------------
def inner_oof(factory, X, y, n_classes, n_inner=3, seed=0):
    """Out-of-fold probabilities on the outer-training set; factory() is re-fitted
    (including its own Prep) on every inner-training split -> no leakage."""
    oof = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        oof[va] = factory().fit(X[tr], y[tr]).predict_proba(X[va])
    return oof


def meta_fit_predict(kind, M_tr, y_tr, M_te, n_classes, n_base):
    if kind == "avg":
        return M_te.reshape(len(M_te), n_base, n_classes).mean(1)
    if kind == "LR":
        clf = LogisticRegression(C=1.0, max_iter=5000)
    elif kind == "LR_C0.1":
        clf = LogisticRegression(C=0.1, max_iter=5000)
    elif kind == "RF":
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)
    elif kind == "NB":
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    else:
        raise ValueError(kind)
    clf.fit(M_tr, y_tr)
    p = np.zeros((len(M_te), n_classes))
    p[:, clf.classes_] = clf.predict_proba(M_te)
    return p
