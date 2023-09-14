
from data_loader import get_train_data, get_test_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

X = np.vstack([X_train,X_test])
Y = np.hstack([y_train,y_test])


colors = ['r' if x == 1 else 'b' for x in Y ]

# 根据 y 对 x 进行 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=10000,  random_state=0)
x_tsne = tsne.fit_transform(X)

# 可视化结果
plt.figure(figsize=(12, 8))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors)
# plt.colorbar()
plt.title('t-SNE Visualization of Data with Labels')
plt.savefig("t-sne_feature.png")