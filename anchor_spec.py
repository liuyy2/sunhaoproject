import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

try:
    import faiss  # 可选加速
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def build_anchors(X, m=1000, seed=0):
    if not 1 <= m <= len(X):
        raise ValueError(f"m must be in [1, {len(X)}], got {m}.")
    km = KMeans(n_clusters=m, n_init=10, random_state=seed)
    km.fit(X)
    return km.cluster_centers_  # [m, d]


def _knn_indices(X, A, s):
    """返回每个 X[i] 到 A 的 s 个最近邻索引和距离平方。"""
    if HAS_FAISS:
        index = faiss.IndexFlatL2(A.shape[1])
        index.add(A.astype(np.float32))
        D, I = index.search(X.astype(np.float32), s)  # D 距离平方
        return I, D
    else:
        dists = pairwise_distances(X, A, metric='sqeuclidean')
        idx = np.argpartition(dists, kth=s-1, axis=1)[:, :s]
        # 取回对应的距离平方
        D = np.take_along_axis(dists, idx, axis=1)
        return idx, D


def build_anchor_graph(X, A, s=3, sigma=None):
    n, m = X.shape[0], A.shape[0]
    idx, D = _knn_indices(X, A, s)

    row_idx = np.repeat(np.arange(n), s)
    col_idx = idx.reshape(-1)
    vals = D.reshape(-1)  # 距离平方

    if sigma is None:
        sigma = np.sqrt(np.median(vals) + 1e-12)

    weights = np.exp(-vals / (2 * sigma**2))
    # 每行归一化
    w = weights.reshape(n, s)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    weights = w.reshape(-1)

    B = csr_matrix((weights, (row_idx, col_idx)), shape=(n, m))
    D_X = np.array(B.sum(axis=1)).ravel()  # [n]
    D_A = np.array(B.sum(axis=0)).ravel()  # [m]
    return B, D_X, D_A, sigma


def anchor_spectral_clustering(X, k, m=1000, s=3, sigma=None, seed=0, return_embedding=False):
    # 1) 锚点
    A = build_anchors(X, m=m, seed=seed)
    # 2) 二部图
    B, D_X, D_A, sigma = build_anchor_graph(X, A, s=s, sigma=sigma)

    # 3) S = D_A^{-1/2} B^T D_X^{-1} B D_A^{-1/2}
    D_X_inv = diags(1.0 / (D_X + 1e-12))
    D_A_inv_sqrt = diags(1.0 / np.sqrt(D_A + 1e-12))
    S = D_A_inv_sqrt @ (B.T @ (D_X_inv @ B)) @ D_A_inv_sqrt

    # 对称化以抑制数值误差
    S = (S + S.T) * 0.5

    # 4) 取最大 k 个特征向量
    eigvals, U = eigsh(S, k=k, which='LA')

    # 5) Y = D_X^{-1/2} B D_A^{-1/2} U
    D_X_inv_sqrt = diags(1.0 / np.sqrt(D_X + 1e-12))
    Y = (D_X_inv_sqrt @ B @ D_A_inv_sqrt) @ U
    # 行归一化
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)

    # 6) KMeans on Y
    km = KMeans(n_clusters=k, n_init=50, random_state=seed)
    labels = km.fit_predict(Y)
    if return_embedding:
        return labels, Y
    return labels
