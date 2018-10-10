import numpy as np

X = np.zeros((10,5))

for batch_slice in gen_batches(n_samples, batch_size):
    X[batch_slice]