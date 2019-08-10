import numpy as np
import matplotlib.pyplot as plt
import utils

from kmeans import kmeans

def gmm1(train_test, save_plots = True, n_clusters = 4, max_it = 200, show_plots = True, print_llk = False):

	data = utils.load_dataset(train_test)

	# Initialization of mu and pi with kmeans
	mu_hat, pi_hat = kmeans('train', save_plots = False, n_clusters = n_clusters, print_results = False)
	
	n_samples = data.shape[0]
	dim = data.shape[1]

	mu_hat = np.transpose(mu_hat)				#[mu] = dim x 1

	sig_hat = 100*np.ones([n_clusters, 1])

	tau = np.zeros([n_samples, n_clusters])

	counter = 0

	llik_old = 0
	llik_new = 10

	while ( (counter < max_it) and np.abs(llik_new - llik_old) > 1e-8):

		llik_old = llik_new

		# E-step
		for i in range(n_samples):
			
			aux = np.zeros(n_clusters)
			
			for l in range(n_clusters):
				
				sig_hat_matrix = sig_hat[l]*np.eye(dim)
				aux[l] = (pi_hat[l] * utils.multivariate_gaussian(np.transpose(data[i, :]), mu_hat[:, l], sig_hat_matrix))						
						
			tau[i, :] = aux / np.sum(aux)

		
		# M-step
		for k in range(n_clusters):
 
			pi_hat[k] = np.sum(tau[:, k]) / n_samples

			# mu_hat
			weighted_samples = np.zeros([1, dim])
			for n in range(n_samples):
				tau_ = tau[n, k]
				weighted_samples += tau_ * data[n, :]
			
			den = np.sum(tau[:, k])
			mu_hat[:, k] = (np.transpose(weighted_samples) / den).reshape(dim)


			# sigma_hat
			weighted_sqnorm = 0
			for n in range(n_samples):

				tau_ = tau[n, k]
				diff = data[n, :].reshape([-1, 1]) - mu_hat[:, k].reshape([-1, 1])			# dim x 1
				sq_norm = np.sum(diff ** 2)				
				weighted_sqnorm += tau_ * sq_norm			


			sig_hat[k, 0] = weighted_sqnorm / (2*den)


		# Log likelihood
		sig_hat_list = [sig_hat[0, 0]*np.identity(dim), sig_hat[1, 0]*np.identity(dim), sig_hat[2, 0]*np.identity(dim), sig_hat[3, 0]*np.identity(dim)]
		llik_new = 0.0		
		for i in range(n_samples):
			for k in range(n_clusters):
				llik_new += tau[i, k] * (np.log(utils.multivariate_gaussian(np.transpose(data[i, :]), mu_hat[:, k], sig_hat_list[k])) + np.log(pi_hat[k]))
		
		llik_new = llik_new / n_samples
	
		counter += 1

	
	if (print_llk):
		print('Centroid for GMM1 on train data')
		print('C1', mu_hat[:, 0])
		print('C2', mu_hat[:, 1])
		print('C3', mu_hat[:, 2])
		print('C4', mu_hat[:, 3])
		print('Log-likelihood for GMM1 on train data :',llik_new)

	if (train_test == 'test'):
		data = utils.load_dataset(train_test)
		n_samples = data.shape[0]

		for i in range(n_samples):
			
			aux = np.zeros(n_clusters)
			
			for l in range(n_clusters):
				
				aux[l] = (pi_hat[l] * utils.multivariate_gaussian(np.transpose(data[i, :]), mu_hat[:, l], sig_hat_list[l]))				
						
			tau[i, :] = aux / np.sum(aux)

		# Log likelihood
		llik_new = 0.0		
		for i in range(n_samples):
			for k in range(n_clusters):
				llik_new += tau[i, k] * (np.log(utils.multivariate_gaussian(np.transpose(data[i, :]), mu_hat[:, k], sig_hat_list[k])) + np.log(pi_hat[k]))
		
		llik_new = llik_new / n_samples

		if (print_llk):
			print('Log-likelihood for GMM1 on test data:',llik_new)
		

	if (show_plots):

		colors = ['c', 'lightskyblue', 'mediumpurple', 'hotpink']

		Z = np.argmax(tau, 1)


		for m in range(n_clusters):
			color = colors[m]
			cluster_samples = data[np.where(Z == m)]
	
			plt.plot(cluster_samples[:, 0], cluster_samples[:, 1], 'o', c = color, label = 'Cluster'+' '+str(m))
			plt.scatter(mu_hat[0, m], mu_hat[1, m], marker='x', s = 100, c = k, linewidths = 5, zorder = 10)

			ellipse_data = utils.plot_ellipse(x_cent = mu_hat[0, m], y_cent = mu_hat[1, m], cov = sig_hat_list[m], mass_level = 0.9)
			plt.plot(ellipse_data[0], ellipse_data[1], c = color)

		

		plt.legend(loc = 'upper left', scatterpoints=1)
		plt.xlabel('Dimension 1')
		plt.ylabel('Dimension 2')

	

		plt.title('Gaussian Mixture Model 1 - '+str(train_test)+' data')

		if (save_plots):
			name = './Figures/gmm1_' + train_test + '.png'
			plt.savefig(name)

		plt.show()

		plt.clf() 	
