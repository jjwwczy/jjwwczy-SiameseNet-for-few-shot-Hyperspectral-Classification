from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os

Feature=np.load('ori_hFeature.npy')
y=np.load('ytest.npy')
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(Feature,y)


plt.figure(figsize=(10, 5))
plt.subplot(121)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y,label="t-SNE")
plt.legend()
plt.savefig('ori_tSNE.png')
plt.show()