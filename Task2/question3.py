import numpy as np
import utils

import matplotlib.pyplot as plt


def question3(dataset_id, train_test, save_plots = False, no_outputs = False):

	x, y = utils.load_dataset(dataset_id, 'train')

	x0 = x[np.where(y[:, 0] == 0)]
	x1 = x[np.where(y[:, 0] == 1)]

	x_bias = np.hstack((np.ones((x.shape[0], 1)), x))

	inner_prod_inv = np.linalg.inv(x_bias.transpose().dot(x_bias))

	# Normal equation solution
	w = (inner_prod_inv.dot(x_bias.transpose())).dot(y)


	feat1 = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 20)
	feat2 = (-w[1]*feat1 - w[0] + 0.5)/w[2]


	if (not no_outputs):
		# Outputs
		print('Parameter vector for dataset',dataset_id,train_test,':',w)
		plt.scatter(x0[:, 0], x0[:, 1], c ='r', marker = 'x', label = 'Class 0')
		plt.scatter(x1[:, 0], x1[:, 1], c ='b', marker = 'x', label = 'Class 1')	
	
		plt.plot(feat1, feat2, c = 'g', label = 'Decision Boundary')	

		plt.legend(loc='lower left', scatterpoints = 1)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')

		if (save_plots):
			name = 'Q3_' + dataset_id + '_' + train_test + '.png'
			plt.savefig(name)

		plt.show()	
		plt.clf()


	#--------- Question 4 ----------------

	if (train_test == 'test'):

		x, y = utils.load_dataset(dataset_id, 'test')
		x_bias = np.hstack((np.ones((x.shape[0], 1)), x))

	correct = 0
	
	for idx in range(x.shape[0]):
		
		sample = x_bias[idx, :]
		label = y[idx, 0]
		
		if (w.transpose().dot(sample) >= 0.5):
			y_tilde = 1.0
		else:
			y_tilde = 0.0

		correct += (label == y_tilde)

	misclassif_error = 1 - correct/x.shape[0]
	
	return misclassif_error



