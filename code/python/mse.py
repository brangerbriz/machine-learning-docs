import numpy as np

a = [0.2, 0.5, 0.7, 10.7, 4.3, 6.5, 80.8]
b = [0.3, 1.0, 0.9, 11.9, 5.0, 7.0, 99.9]

def mse(a, b):
	error = 0.0
	for i in range(len(a)):
		error += (b[i] - a[i]) ** 2
	return error / len(a)

def mse_numpy(a, b):
	return ((a - b) ** 2).mean(axis=None)

print('mse: {}'.format(mse(a, b)))
print('mse_numpy: {}'.format(mse_numpy(np.array(a), np.array(b))))