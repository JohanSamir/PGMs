import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt



def normalize(x):
	mean = np.mean(x, axis = 0)
	std = np.std(x, axis = 0)

	normalized = (x - mean)/(std + 1e-10)

	return normalized

def load_dataset(train_test = 'train', path = './hwk4data/'):
	
	filename = path + 'EMGaussian' + '_' + train_test + '.txt'

	data = []

	with open(filename) as file_:
		
		reader = file_.readlines()
		
		for row in reader:
			data.append([float(sample) for sample in row.split(" ")])

	
	data = np.asarray(data)

	return data 

def sigmoid(x):
	
	sig = 1.0/ (1.0 + np.exp(-x) )

	return sig 

def multivariate_gaussian(x, mu, cov):
	
	d = mu.shape[0]

	#mu = mu.reshape([-1, 1])

	p_x = (1/(np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov)))) * np.exp(- (1/2) * ( np.transpose(x - mu).dot(np.linalg.inv(cov).dot((x - mu))) ) )

	return p_x	

def multivariate_gaussian_all_z(x, all_mu, cov_list):
	
	d = all_mu.shape[0]
	k = all_mu.shape[1]

	all_p_x = np.zeros([k, 1])

	for idx in range(k):	

		all_p_x[idx, 0] = multivariate_gaussian(x, all_mu[:, idx], cov_list[idx])

	return all_p_x

def alpha_log(x, pi, mu, sigma_list, A, n_states):

	time_steps = x.shape[0]
	n_states = n_states

	alpha_log = np.zeros([n_states, time_steps])
	alpha_log[:, 0] = np.squeeze(np.log(pi) + np.log(multivariate_gaussian_all_z(x[0, :], mu, sigma_list)))

	for t in range(1, time_steps):
		for i in range(n_states):
			p_xz = multivariate_gaussian(x[t, :], mu[:, i], sigma_list[i])
			aux = np.zeros(n_states)
			for j in range(n_states):
				aux[j] = np.log(p_xz) + np.log(A[i, j]) + alpha_log[j, t-1]
			b = np.max(aux)
			alpha_log[i, t] = b + np.log(np.sum(np.exp(aux - b)))

	return alpha_log

def beta_log(x, pi, mu, sigma_list, A, n_states):

	time_steps = x.shape[0]
	n_states = n_states 

	beta_log = np.zeros([n_states, time_steps])
	beta_log[:, time_steps-1] = 1.0

	for t in range(1, time_steps):
		idx = time_steps - t - 1
		for i in range(n_states):
			aux = np.zeros(n_states)
			for j in range(n_states):
				aux[j] = np.log(A[j, i]) + np.log(multivariate_gaussian(x[idx+1, :], mu[:, j], sigma_list[j])) + beta_log[j, idx+1]			# Checar A[zt, zt1]
			b = np.max(aux)	
			beta_log[i, idx] = b + np.log(np.sum(np.exp(aux - b)))

	return beta_log

def gamma_log(alpha_log, beta_log):

	time_steps = alpha_log.shape[1]
	n_states = alpha_log.shape[0]
	gamma_log = np.zeros([n_states, time_steps])
	for t in range(time_steps):
		for k in range(n_states):
			aux = np.zeros(n_states)
			for j in range(n_states):
				aux[j] = alpha_log[j, t] + beta_log[j, t]
			b = np.max(aux)
			den = b + np.log(np.sum(np.exp(aux - b)))
			gamma_log[k, t] = (alpha_log[k, t] + beta_log[k, t]) - den
	
	return gamma_log 


def log_likelihood(n_states, time_steps, gamma, csi, data, A, pi, mu, sigma_list):
	
	llik1 = 0.0
	llik2 = 0.0
	llik3 = 0.0

	for r in range(n_states):
		llik1 = gamma[r, 0] * np.log(pi[r] + 1e-100) + llik1

	for t in range(time_steps):
		for l in range(n_states):
			llik2 = (gamma[l, t] * np.log(multivariate_gaussian(data[t, :], mu[:, l], sigma_list[l]))) + llik2

	for t in range(time_steps-1):
		for l in range(n_states):
			for m in range(n_states):
				llik3 = csi[t, l, m] * np.log(A[l, m] + 1e-100) + llik3

	llik = llik1 + llik2 + llik3

	return llik


def plot_ellipse(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1e3, ax=None, plot_kwargs=None, cov=None, mass_level=0.68):
	# Get Ellipse Properties from cov matrix
	eig_vec, eig_val, u = np.linalg.svd(cov)
	# Make sure 0th eigenvector has positive x-coordinate
	if eig_vec[0][0] < 0:
		eig_vec[0] *= -1
	semimaj = np.sqrt(eig_val[0])
	semimin = np.sqrt(eig_val[1])
	distances = np.linspace(0,20,20001)
	chi2_cdf = chi2.cdf(distances,df=2)
	multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
	semimaj *= multiplier
	semimin *= multiplier
	phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
	if eig_vec[0][1] < 0 and phi > 0:
		phi *= -1

	# Generate data for ellipse structure
	theta = np.linspace(0, 2*np.pi, theta_num)
	r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	data = np.array([x,y])
	S = np.array([[semimaj, 0], [0, semimin]])
	R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
	T = np.dot(R,S)
	data = np.dot(T, data)
	data[0] += x_cent
	data[1] += y_cent

	return data
