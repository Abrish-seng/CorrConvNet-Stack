import numpy as np, pandas as pd
def load(name):
    if name == "11tumors":
        d = pd.read_csv("/home/claude/data/ML_DL_microArrays/data11tumors2.csv")
        y = d["Class"].values.astype(int); X = d.drop(columns=["Class"]).values.astype(np.float64)
        names = [f"att{i}" for i in range(1, X.shape[1]+1)]
        classes = [str(c) for c in sorted(set(y))]
    elif name == "leukemia":
        d = pd.read_csv("/home/claude/data2/aml/data/Leukemia_GSE9476.csv")
        lab = d["type"].values; classes = sorted(set(lab))
        y = np.array([classes.index(v) for v in lab]); X = d.drop(columns=["samples","type"]).values.astype(np.float64)
        names = list(d.columns[2:])
    return X, y, names, classes
