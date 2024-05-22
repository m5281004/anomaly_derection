import numpy as np

def pca_svd(X):
    """
    SVD を用いて主成分分析を実行します. 

    パラメータ
    ----------
    X: データ行列

    戻り値
    -------
    P: ローディング行列
    t: 主成分得点行列
    """

    # 行列を特異値分します
    _, _, P = np.linalg.svd(X)

    # 主成分得点を計算します
    t = X @ P.T
    return P, t