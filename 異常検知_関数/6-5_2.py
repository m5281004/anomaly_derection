##
## 藤原(2022) 第6章異常検知 マネ
## p234-, 関連する部分はそれより前に出ていることも

##
## ---- ライブラリ
##

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


import torch 
from torch import nn, optim
from torch.utils.data import DataLoader

# 自作関数
from autoencorder import Autoencoder, AE_T2RE
from mspc import mspc_CL, mspc_ref, mspc_T2Q, cont_T2Q
from scale import autoscale, scaling


##
## ------------------------6.5 データの読み込み ------------------------
##

# 正常データ
train_data = pd.read_csv('normal_data.csv')
# 異常データ  対象とする異常によって読み込むファイルを変更してください．
faulty_data = pd.read_csv('idv 14_data.csv') # idv 1_data.csv


##
## ------------------------6.6 LOFでの異常検知------------------------
##

# 正常データを用いたモデルの学習
model_LOF = LocalOutlierFactor(n_neighbors = 50, novelty = True, contamination = 0.01) # novelty: 未知のデータに対して適用する場合True。n_neiborsとcontaminationはそれぞれ調整する必要
model_LOF.fit(train_data)

# 管理限界の取得
CL_lof = model_LOF.offset_ 
# 異常データのLOF スコアの計算
score_lof = model_LOF.score_samples(faulty_data) # 960。負の値で出力されるらしい。管理限界 CL_lof より小さい場合に異常と判定

##
## ------------------------ 6.7 iForestでの異常検知 ------------------------
##

# 正常データを用いたモデルの学習
model_IFOREST = IsolationForest(contamination = 0.05)
model_IFOREST.fit(train_data)

# 管理限界の取得
CL_if = model_IFOREST.offset_
# 異常データのiForest スコアの計算
score_if = model_IFOREST.score_samples(faulty_data)


##
## ------------------------6.8 MSPCでの異常検知------------------------
##

# 正常データを用いたモデルの学習
meanX, stdX, U, S, V = mspc_ref(train_data, numPC = 17) # numPCは主成分の数。データの次元に合わせて変える必要。この例では53次元を17次元に縮約。

# 管理限界の決定
T2_train, Q_train = mspc_T2Q(train_data, meanX, stdX, U, S, V) 
# 異常データのT2 統計量とQ 統計量の計算
T2_mspc, Q_mspc = mspc_T2Q(faulty_data, meanX, stdX, U, S, V)


##
## ------------------------　6.10 AEモデルの学習------------------------
##

# ハイパーパラメータの設定
z_dim = 17 # 中間層の次元

# インスタンスの作成
model_AE = Autoencoder(z_dim = z_dim) 

criterion = nn.MSELoss() # 誤差関数(平均二乗誤差)
optimizer = torch.optim.Adam(model_AE.parameters(), lr = 0.0001) # オプティマイザ
num_epochs = 110 
batch_size = 20 

# 学習用データの前処理
train_data, mean_train, std_train = autoscale(train_data)
train_data = train_data.astype('float32')
train_data = train_data.values
train_data = torch.tensor(train_data) # Tensor 型への変換
dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)



# 学習ループ
for epoch in range(num_epochs):
    for data in dataloader:
        xhat, z = model_AE.forward(data)
        loss = criterion(xhat, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

## ------------------------ 6.11 z_bar と S_z の算出 ------------------------
x_hat, z = model_AE.forward(train_data)
z = z.detach().numpy()
z_bar = np.average(z, axis = 0)
z_bar = np.reshape(z_bar, (len(z_bar), 1))
S_z = np.cov(z.T)

## ------------------------ 6.13 AEでの管理限界の設定------------------------
# 管理限界の決定
T2_train, RE_train = AE_T2RE(train_data, z_bar, S_z, model_AE)
# MSPCの管理限界関数をAEに流用
CL_T2_AE, CL_RE_AE = mspc_CL(T2_train, RE_train, train_data, alpha = 0.99)

## ------------------------ 6.14 AEでの異常検知 ------------------------
# 前処理
faulty_data = scaling(faulty_data, mean_train, std_train)
faulty_data = faulty_data.astype('float32')
faulty_data = faulty_data.values
faulty_data = torch.tensor(faulty_data)

# T2 統計量, RE の計算
T2_AE, RE_AE = AE_T2RE(faulty_data, z_bar, S_z, model_AE)


##
## ------------------------ 6.15 異常検知結果のプロット ------------------------
##

plt.clf() # すでに書かれていた図を消す

# LOF(赤)
plt.plot(
  list(range(1, 961)),
  abs(score_lof),
  "r") # LOF では絶対値を計算
plt.hlines(abs(CL_lof), 1, 960, "r", linestyles = 'dashed') # 管理限界(点線)より上で異常

# iForest(青)
plt.plot(
  list(range(1, 961)), 
  abs(score_if)*2, # みやすいようにスケール調整 
  "b")  # iForest では絶対値を計算
plt.hlines(abs(CL_if)*2,  1, 960, "b", linestyles = 'dashed') # 管理限界(点線)より上で異常

# 異常発生時刻
plt.vlines(160, 0, 2.5, "g", linestyles = "dashed") 

plt.ylabel('Anomaly Score'); plt.show()


## ------------------------ 6.16 寄与プロットによる異常診断------------------------

# 寄与の計算
cont_T2, cont_Q = cont_T2Q(np.array(faulty_data), np.array(meanX), np.array(stdX), U, S, V) 

# 異常発生後100サンプルの寄与の平均を計算
fault_cont_T2 = np.average(cont_T2[160:260, :], axis = 0)
fault_cont_Q  = np.average( cont_Q[160:260, :], axis = 0)

# 寄与の上位6番目までをプロット
#   T2ベース
fault_cont_T2_ser = pd.Series(fault_cont_T2,index = meanX.index)
fault_cont_T2_ser = fault_cont_T2_ser.sort_values(ascending=False)

plt.figure(tight_layout=True)
fault_cont_T2_ser.iloc[0:6].plot.bar()
plt.show()

#   Qベース
fault_cont_Q_ser = pd.Series(fault_cont_Q, index= meanX.index).sort_values(ascending=False)

plt.figure(tight_layout=True)
fault_cont_Q_ser.iloc[0:6].plot.bar()
plt.show()

