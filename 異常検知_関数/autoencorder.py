
# 
# プログラム6.9 (p236) AEのクラス定義
# 

import numpy as np
from torch import nn

# クラス定義
class Autoencoder(nn.Module):
    def __init__(self, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(52, z_dim),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 52),
            nn.ReLU(True)
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

# 
# プログラム6.12 (p238) AEでのT2統計量とREの計算
# 

# T2とREを取得する関数
def AE_T2RE(X, z_bar, S_z, model):
  xhat_tensor, z_tensor = model.forward(X)
  z = z_tensor.detach().numpy()
  xhat = xhat_tensor.detach().numpy()

  T2_AE = np.empty(len(X))
  RE_AE = np.empty(len(X))

  for i in range(len(z)):
    z_vec = np.reshape(z[i], (len(z[i]), 1))
    T2 = (z_vec - z_bar).T @ np.linalg.inv(S_z) @ (z_vec - z_bar)
    RE = (X[i] - xhat[i])**2

    T2_AE[i] = T2[0]
    RE_AE[i] = RE[0]

  return T2_AE, RE_AE
