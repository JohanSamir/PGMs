import numpy as np
import utils

import matplotlib.pyplot as plt

def question1(dataset_id, train_test, save_plots = False, no_outputs = False):

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

	Sigma_hat = 1 / (n1 + n0) * (Sigma0 + Sigma1)


	beta = np.linalg.inv(Sigma_hat).dot(mean1_hat - mean0_hat)
	gamma = -0.5*(mean1_hat - mean0_hat).transpose().dot(np.linalg.inv(Sigma_hat)).dot((mean1_hat + mean0_hat)) + np.log(pi_hat/(1 - pi_hat))

	feat1 = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 20)
	feat2 = (-beta[0]*feat1 - gamma)/beta[1]

	
	if (not no_outputs):
		# Outputs
		print('Dataset',dataset_id, train_test)
		print('MLE for pi ', pi_hat)
		print('MLE for mean0 ', mean0_hat)
		print('MLE for mean1 ', mean1_hat)
		print('MLE for Sigma ', Sigma_hat)
		plt.scatter(x0[:, 0], x0[:, 1], c ='r', marker = 'x', label = 'Class 0')
		plt.scatter(x1[:, 0], x1[:, 1], c ='b', marker = 'x', label = 'Class 1')	
		plt.plot(feat1, feat2.flatten(), c = 'g', label = 'Decision Boundary')


		plt.legend(loc='lower left', scatterpoints = 1)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')

		if (save_plots):
			name = 'Q1_' + dataset_id + '_' + train_test + '.png'
			plt.savefig(name)


		plt.show()	
		plt.clf()


	#-------------Question 4 ---------------

	if (train_test == 'test'):

		x, y = utils.load_dataset(dataset_id, 'test')

	correct = 0
	
	for idx in range(x.shape[0]):
		
		sample = x[idx, :]
		label = y[idx, 0]
		
		if ((utils.sigmoid(beta.transpose().dot(sample) + gamma)) >= 0.5):
			y_tilde = 1.0
		else:
			y_tilde = 0.0

		correct += (label == y_tilde)

	misclassif_error = 1 - correct/x.shape[0]
	
	return misclassif_error

