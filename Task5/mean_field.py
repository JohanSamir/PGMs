'''
Mean field variational inference for Ising model

'''

import numpy as np
import utils
import matplotlib.pyplot as plt

#np.random.seed(10)

def mean_field(init = 3):

	n_vert = 7
	n_i = -1
	n_st = 0.5


	kl_qp = []

	max_it = 10000
	d = float('inf')
	counter = 0

	# Inititalization
	if (init == 1):
		tau = np.random.rand(n_vert, n_vert)

	elif (init == 2):
		tau = np.random.normal(0.3, 0.01, [n_vert, n_vert])

	elif (init == 3):
		tau = np.random.normal(0.7, 0.1, [n_vert, n_vert])		# tau = np.random.normal(1, 1, [n_vert, n_vert]) weird result!!

	elif (init == 4):
		tau = np.random.normal(0.5, 0.5, [n_vert, n_vert])

	tau_old = np.zeros([n_vert, n_vert])

	while (counter < max_it and d >= 0.00001):

		kl_qp_ = 0	
		for i in range(n_vert):
			for j in range(n_vert):

				idx = i*7 + j + 1

				sum_ = n_st * ( tau[(i-1) % n_vert, j] + tau[(i+1) % n_vert , j] + tau[i, (j-1) % n_vert] + tau[i, (j+1) % n_vert] )

				potential = (n_i**(idx)) + sum_

				tau[i, j] = utils.sigmoid(potential)
				
				kl_qp_ += tau[i, j] * np.log(tau[i, j]) + (1 - tau[i, j]) * np.log(1 - tau[i, j]) - (n_i**idx) * tau[i, j]

		for i in range(n_vert):
			for j in range(n_vert):
				kl_qp_ -= 0.5 * tau[i, j]* (tau[(i+1) % n_vert , j] + tau[i, (j+1) % n_vert]) 

		d = utils.dist(tau, tau_old)

		tau_old = tau.copy()
		counter += 1

		kl_qp.append(kl_qp_)

	plt.plot(kl_qp)
	plt.title('KL(q||p) - logZp for Mean Field Variationl Inference - Init. '+ str(init))
	plt.ylabel('KL(q||p) - logZp')
	plt.xlabel('Iterations')
	name = './Figures/kl_qp_init' + str(init) +'.png'
	plt.savefig(name)
	plt.show()
	plt.clf()

	return(tau)
