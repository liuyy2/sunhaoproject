import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from anchor_spec import anchor_spectral_clustering


def clustering_acc(y_true, y_pred):
    y_true = y_true.astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / y_pred.size


def main(features_path, k, m, s, seed):
    data = np.load(features_path)
    X, y = data['X'], data['y']

    labels_anchor = anchor_spectral_clustering(X, k=k, m=m, s=s, seed=seed)
    # save_embed use in painting
    #labels_anchor, Y = anchor_spectral_clustering(X, k=k, m=m, s=s, seed=seed, return_embedding=True)
    try:
        km = KMeans(n_clusters=k, n_init='auto', random_state=seed)
    except TypeError:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels_km = km.fit_predict(X)

    def report(name, yp):
        ari = adjusted_rand_score(y, yp)
        nmi = normalized_mutual_info_score(y, yp)
        acc = clustering_acc(y, yp)
        print(f"{name:16s} | ARI={ari:.4f}  NMI={nmi:.4f}  ACC={acc:.4f}")

    report('AnchorSpectral', labels_anchor)
    report('KMeans-Direct', labels_km)
    #if save_embed and save_embed.lower() != 'none':
        #np.savez(save_embed, X=Y, y=y, pred=labels_anchor)  # 注意：这里用 X=Y 是为了复用 vis_2d.py
        #print(f"[OK] Saved spectral embedding to {save_embed} | Y={Y.shape}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_path', default='features_test.npz')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--s', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    #ap.add_argument('--save_embed', default='none')
    args = ap.parse_args()
    main(**vars(args))