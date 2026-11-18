"""1D Grad-CAM++, gradient x input, LIME, and deletion/insertion faithfulness."""
import numpy as np
import torch
import torch.nn.functional as F
from corrconv import predict_cnn


def gradcampp(models, Z, cls):
    """Grad-CAM++ (Chattopadhay et al. 2018, closed form with exp(score)) on the
    final conv feature map, linearly up-sampled to gene resolution; averaged over seeds."""
    X = torch.tensor(Z)
    L = Z.shape[1]
    maps = []
    for m in models:
        m.eval()
        A = m.features(X)
        logits = m.forward_from_features(A)
        s = logits[torch.arange(len(X)), torch.tensor(cls)]
        g = torch.autograd.grad(s.sum(), A)[0]
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + A.sum(-1, keepdim=True) * g3
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alpha * F.relu(g)).sum(-1, keepdim=True)
        cam = F.relu((w * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=L, mode="linear", align_corners=False)[:, 0]
        cam = cam / (cam.amax(-1, keepdim=True) + 1e-12)
        maps.append(cam.detach().numpy())
    return np.mean(maps, 0), np.stack(maps)


def grad_x_input(models, Z, cls):
    X = torch.tensor(Z, requires_grad=True)
    tot = 0
    for m in models:
        s = m(X)[torch.arange(len(Z)), torch.tensor(cls)]
        tot = tot + torch.autograd.grad(s.sum(), X)[0]
    return np.abs((tot / len(models) * X).detach().numpy())


def lime_weights(models, Ztr, Zte, cls, n_samples=1000, seed=0):
    from lime.lime_tabular import LimeTabularExplainer
    exp = LimeTabularExplainer(Ztr, mode="classification", discretize_continuous=False,
                               random_state=seed, feature_selection="none")
    fn = lambda A: predict_cnn(models, A.astype(np.float32))
    W = np.zeros_like(Zte)
    for i in range(len(Zte)):
        e = exp.explain_instance(Zte[i], fn, labels=(int(cls[i]),), num_features=Zte.shape[1],
                                 num_samples=n_samples)
        for j, v in e.as_map()[int(cls[i])]:
            W[i, j] = v
    return W


KS = [1, 2, 5, 10, 20, 50, 100, 200, 400]


def deletion_insertion(models, Z, cls, scores, ks=KS):
    """Deletion: set the top-k attributed genes to the training mean (0 after z-scoring)
    and record p(class). Insertion: keep only the top-k genes (others = 0).
    scores: (n, L) importance (higher = more important)."""
    n = len(Z)
    order = np.argsort(-scores, 1)
    p0 = predict_cnn(models, Z)[np.arange(n), cls]
    dele, ins = np.zeros((n, len(ks))), np.zeros((n, len(ks)))
    for j, k in enumerate(ks):
        D, I = Z.copy(), np.zeros_like(Z)
        for i in range(n):
            D[i, order[i, :k]] = 0.0
            I[i, order[i, :k]] = Z[i, order[i, :k]]
        dele[:, j] = predict_cnn(models, D)[np.arange(n), cls]
        ins[:, j] = predict_cnn(models, I)[np.arange(n), cls]
    return p0, dele, ins
