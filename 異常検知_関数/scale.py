import numpy as np

def autoscale(X):
    """
    データ行列を標準化します

    パラメータ
    ----------
    X: データ行列

    戻り値
    -------
    Xscale: 標準化後のデータ行列
    meanX: 平均値ベクトル
    stdX: 標準偏差ベクトル
    """
    meanX = np.mean(X, axis = 0)
    stdX = np.std(X, axis = 0, ddof = 1)
    Xscale = (X - meanX) / stdX
    return Xscale, meanX, stdX


def scaling(x, meanX, stdX):
    """
    データ行列の平均と標準偏差からサンプルを標準化します

    パラメータ
    ----------
    x: 標準化したいサンプル
    meanX: 平均値ベクトル
    stdX: 標準偏差ベクトル

    戻り値
    -------
    xscale: 標準化後のサンプル
    """
    xscale = (x - meanX) / stdX
    return xscale


def rescaling(xscale, meanX, stdX):
    """
    標準化されたサンプルを元のスケールに戻します

    パラメータ
    ----------
    xscale: 標準化後のサンプル
    meanX: 平均値ベクトル
    stdX: 標準偏差ベクトル

    戻り値
    -------
    x: 元のスケールのサンプル
    """
    x = np.multiply(stdX , xscale) + meanX
    return x
  
