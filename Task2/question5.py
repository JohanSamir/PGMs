import numpy as np
import utils

import matplotlib.pyplot as plt

def question5(dataset_id, train_test, save_plots = False, no_outputs = False):

	x, y = utils.load_dataset(dataset_id, 'train')

	x0 = x[np.where(y[:, 0] == 0)]
	x1 = x[np.where(y[:, 0] == 1)]

	n0 = x0.shape[0]
	n1 = x1.shape[0] 

	pi_hat = n1 / (n1 + n0) 

	mean0_hat = np.mean(x0, axis = 0).reshape([-1, 1])
	mean1_hat = np.mean(x1, axis = 0).reshape([-1, 1])

	Sigma0 = np.zeros([x0.shape[1], x0.shape[1]])
	Sigma1 = np.zeros([x1.shape[1], x1.shape[1]])	

	for sample in x0:
		
		sample = sample.reshape([-1, 1])
		Sigma0 += (sample - mean0_hat).dot((sample - mean0_hat).transpose())


	for sample in x1:

		sample = sample.reshape([-1, 1])
		Sigma1 += (sample - mean1_hat).dot((sample - mean1_hat).transpose())


	Sigma0_hat = (1 / n0) * Sigma0
	Sigma1_hat = (1 / n1) * Sigma1

	s0_det = np.linalg.det(Sigma0_hat)
	s1_det = np.linalg.det(Sigma1_hat)

	if (not no_outputs):
		# Outputs
		print('Dataset',dataset_id, train_test)
		print('MLE for pi ', pi_hat)
		print('MLE for mean0 ', mean0_hat)
		print('MLE for mean1 ', mean1_hat)
		print('MLE for Sigma0 ', Sigma0_hat)
		print('MLE for Sigma1 ', Sigma1_hat)
		plt.scatter(x0[:, 0], x0[:, 1], c ='r', marker = 'x', label = 'Class 0')
		plt.scatter(x1[:, 0], x1[:, 1], c ='b', marker = 'x', label = 'Class 1')	


		x_h = np.linspace(np.min(x[:, 0]),np.max(x[:, 0]), 50)
		x_v = np.linspace(np.min(x[:, 1]),np.max(x[:, 1]), 50)


		x_h, x_v = np.meshgrid(x_h, x_v)

		z = np.zeros(x_h.shape)

		for i in range(x_h.shape[0]):
			for j in range(x_h.shape[1]):
				x_ = np.asarray([x_h[i][j], x_v[i][j]]).reshape([2,1])
				z[i][j] = -0.5*(x_ - mean0_hat).transpose().dot(np.linalg.inv(Sigma0_hat)).dot(x_ - mean0_hat) + 0.5*(x_ - mean1_hat).transpose().dot(np.linalg.inv(Sigma1_hat)).dot(x_ - mean1_hat) + np.sqrt(s1_det/s0_det) + np.log((1 - pi_hat)/pi_hat)

		plt.contour(x_h, x_v,z,[0], colors = 'g', label = 'Decision Boundary')

		plt.legend(loc='lower left', scatterpoints = 1)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')

		if (save_plots):
			name = 'Q5_' + dataset_id + '_' + train_test + '.png'
			plt.savefig(name)


		plt.show()	
		plt.clf()

	
	#-------------Item C ---------------

	correct = 0

	if (train_test == 'test'):

		x, y = utils.load_dataset(dataset_id, 'test')
	
	for idx in range(x.shape[0]):
		
		sample = x[idx, :].reshape([-1, 1])
		label = y[idx, 0]

		aux0 = 0.5*(((sample - mean0_hat).transpose().dot(np.linalg.inv(Sigma0_hat))).dot(sample - mean0_hat))

		aux1 = 0.5*(((sample - mean1_hat).transpose().dot(np.linalg.inv(Sigma1_hat))).dot(sample - mean1_hat))

		aux_pi = 0.5*np.log(s1_det/s0_det) + np.log((1 - pi_hat)/pi_hat)

		value = -aux1 + aux0 + aux_pi
		
		if (utils.sigmoid(value) >= 0.5):
			y_tilde = 1.0
		else:
			y_tilde = 0.0

		correct += (label == y_tilde)

	misclassif_error = 1 - correct/x.shape[0]

	print('Misclassification error for QDA on dataset',dataset_id,train_test,':',misclassif_error)	
