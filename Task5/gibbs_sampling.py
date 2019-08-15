'''
Gibbs sampling for Ising model

'''

import numpy as np
import utils
np.random.seed(10)


def gibbs_sampling():

	n_vert = 7
	n_i = -1
	n_st = 0.5

	n_rep = 10
	burnin = 1000
	epochs = 5000

	moments_list = [] 

	for rep in range(n_rep):

		samples = np.random.binomial(n = 1, p = np.random.rand(n_vert, n_vert))
		samples_accum = np.zeros([n_vert, n_vert])

		for it in range(burnin):

			for i in range(n_vert):
				for j in range(n_vert):

					idx = i*7 + j + 1

					sum_ = n_st * ( samples[(i-1) % 7, j] + samples[(i+1) % 7 , j] + samples[i, (j-1) % 7] + samples[i, (j+1) % 7] )

					potential = (n_i**(idx)) + sum_
	
					prob = utils.sigmoid(potential)
			
					u = np.random.rand()	

					if (u <= prob):
						samples[i, j] = 1.0
					else:
						samples[i, j] = 0.0
		 
		for epc in range(epochs):

			for i in range(n_vert):
				for j in range(n_vert):

					idx = i*7 + j + 1

					sum_ = n_st * ( samples[(i-1) % 7, j] + samples[(i+1) % 7 , j] + samples[i, (j-1) % 7] + samples[i, (j+1) % 7] )

					potential = (n_i**(idx)) + sum_
	
					prob = utils.sigmoid(potential)
			
					u = np.random.rand()	

					if (u <= prob):
						samples[i, j] = 1.0
					else:
						samples[i, j] = 0.0

			samples_accum += samples
	
		moments_list.append(samples_accum/epochs)

	moments = np.asarray(moments_list)

	std_dev = np.sqrt(np.var(moments, axis = 0))
	mu_hat = np.mean(moments, axis = 0)
	
	return(std_dev, mu_hat)


	
		
