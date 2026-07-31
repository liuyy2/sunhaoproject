# vis_2d.py
# for 特征空间可视化
import argparse, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def stratified_subsample(X, y, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    classes = np.unique(y)
    per = max(1, n // len(classes))
    for c in classes:
        inds = np.where(y == c)[0]
        take = min(per, len(inds))
        idx.append(rng.choice(inds, size=take, replace=False))
    idx = np.concatenate(idx)
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return X[idx], y[idx]

def run_tsne(Z, seed=0):
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=30, random_state=seed)
    return tsne.fit_transform(Z)

def run_umap(Z, seed=0):
    try:
        import umap
    except ImportError:
        raise SystemExit("UMAP not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.1,
                        metric="euclidean", random_state=seed)
    return reducer.fit_transform(Z)

def plot_2d(E, y, title, out_png):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(E[:, 0], E[:, 1], c=y, s=6, alpha=0.85, cmap="tab10")
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("[OK] saved:", out_png)

def main(npz_path, method, n_samples, seed, out_png, title):
    d = np.load(npz_path)
    #X, y = d["X"], d["y"].astype(int)
    X = d['X']
    if args.color == 'pred':
        y = d['pred']
    else:
        y = d['y']

    # 采样（10k 全跑 t-SNE 可能慢）
    Xs, ys = stratified_subsample(X, y, n=n_samples, seed=seed)

    # 先 PCA 到 50 维，提速+降噪（UMAP/tSNE 都建议）
    Z = PCA(n_components=min(50, Xs.shape[1]), random_state=seed).fit_transform(Xs)

    if method == "tsne":
        E = run_tsne(Z, seed)
    else:
        E = run_umap(Z, seed)

    plot_2d(E, ys, title, out_png)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument('--color', default='y', choices=['y', 'pred'])
    ap.add_argument("--method", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    main(args.npz, args.method, args.n_samples, args.seed, args.out, args.title)