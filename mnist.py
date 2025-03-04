import numpy as np


from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", as_frame=False, return_X_y=True)

np.savetxt('data/mnist-images.txt', X, fmt='%d')
np.savetxt('data/mnist-labels.txt', y, fmt='%s')
