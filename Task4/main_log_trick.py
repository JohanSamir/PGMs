import numpy as np
import utils
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


np.random.seed(12345)

save_plots = True	
data_train = utils.load_dataset('train')
data_test = utils.load_dataset('test')
	
time_steps = data_train.shape[0]
dim = data_train.shape[1]
n_states = 4

# Questions 1 and 2 

A = np.ones([4, 4])
A = A * (1/6)
A[0, 0] = 1/2
A[1, 1] = 1/2
A[2, 2] = 1/2
A[3, 3] = 1/2 

mu = np.array([(-2.0344, 3.9779, 3.8007, -3.0620), (4.1726, 3.7735, -3.7972, -3.5345)])

sigma_list = [ np.array([(2.9044, 0.2066), (0.2066, 2.7562)]), np.array([(0.2104, 0.2904), (0.2904, 12.2392)]), np.array([(0.9213, 0.0574), (0.0574, 1.8680)]), np.array([(6.2414, 6.0502), (6.0542, 6.1825)])]

pi = np.ones([n_states, 1])
pi /= np.sum(pi)

# Alpha
alpha_log = utils.alpha_log(data_test, pi, mu, sigma_list, A, n_states)

# Beta
beta_log = utils.beta_log(data_test, pi, mu, sigma_list, A, n_states)

# Gamma (smoothing disribution)
gamma_log = utils.gamma_log(alpha_log, beta_log)

gamma = np.exp(gamma_log)

# Csi (pair marginals)
csi_log = np.zeros([time_steps-1, n_states, n_states])
for t in range(time_steps-1):
	aux = np.zeros([n_states, n_states])
	for m in range(n_states):
		for l in range(n_states):
			aux[m, l] = alpha_log[m, t] + beta_log[l, t+1] + np.log(A[l, m]) + np.log(utils.multivariate_gaussian(data_train[t+1, :], mu[:, l], sigma_list[l]))	
	b = np.max(aux)
	den = b + np.log(np.sum(np.exp(aux - b)))
	
	for i in range(n_states):
		for j in range(n_states):
			csi_log[t, j, i] = alpha_log[i, t] + beta_log[j, t+1] + np.log(A[j, i]) + np.log(utils.multivariate_gaussian(data_train[t+1, :], mu[:, j], sigma_list[j])) - den

csi = np.exp(csi_log)

plt.subplot(411)
plt.plot(gamma[0, 0:100], 'c')
plt.title("$p(z_t|x_1, \ldots , x_T)$ - HMM (Fake parameters) - Test data")
plt.ylabel('State 0')

plt.subplot(412)
plt.plot(gamma[1, 0:100], 'lightskyblue')
plt.ylabel('State 1')

plt.subplot(413)
plt.plot(gamma[2, 0:100], 'mediumpurple')
plt.ylabel('State 2')

plt.subplot(414)
plt.plot(gamma[3, 0:100], 'hotpink')
plt.ylabel('State 3')
plt.xlabel('Time steps')


if (save_plots):
	name = './Figures/fake_params.png'
	plt.savefig(name)

plt.show()
plt.clf()

# Questions 3, 4 and 5

max_it = 50	
counter = 0

llik_old = 0
llik_new = 10

llik_list = []
llik_list_test = []

while (counter < max_it and ((np.abs(llik_new - llik_old)) > 1e-8)):

	llik_old = llik_new

	# E-step:
	
	# Alpha
	alpha_log = utils.alpha_log(data_train, pi, mu, sigma_list, A, n_states)
	
	# Beta
	beta_log = utils.beta_log(data_train, pi, mu, sigma_list, A, n_states)

	# Gamma
	gamma_log = utils.gamma_log(alpha_log, beta_log) 

	gamma = np.exp(gamma_log)

	# Csi
	csi_log = np.zeros([time_steps-1, n_states, n_states])
	for t in range(time_steps-1):
		aux = np.zeros([n_states, n_states])
		for m in range(n_states):
			for l in range(n_states):
				aux[m, l] = alpha_log[m, t] + beta_log[l, t+1] + np.log(A[l, m]) + np.log(utils.multivariate_gaussian(data_train[t+1, :], mu[:, l], sigma_list[l]))	
		b = np.max(aux)
		den = b + np.log(np.sum(np.exp(aux - b)))
		
		for i in range(n_states):
			for j in range(n_states):
				csi_log[t, j, i] = alpha_log[i, t] + beta_log[j, t+1] + np.log(A[j, i]) + np.log(utils.multivariate_gaussian(data_train[t+1, :], mu[:, j], sigma_list[j])) - den

	csi = np.exp(csi_log)

	# M-step:

	# pi
	pi = gamma[:, 0].reshape([-1, 1])		# Is it equal to tau_tk[:, 0]?

	# Transition matrix
	for l in range(n_states):
		for m in range(n_states):
			A[l, m] = np.sum(csi[0:-1, l, m]) / np.sum(csi[0:-1, :, m])	##### [zt,zt-1]

	for j in range(n_states):
		# Mean
		weighted_samples = np.zeros([1, dim])
		den = 0
		for t in range(time_steps):
			gamma_ = gamma[j, t]
			weighted_samples += gamma_ * data_train[t, :]

			den += gamma[j, t]
		mu[:, j] = (np.transpose(weighted_samples) / den).reshape(dim)

		# Sigma
		weighted_diff = np.zeros([dim, dim])
		for t in range(time_steps):
			gamma_ = gamma[j, t]
			diff = data_train[t, :].reshape([-1, 1]) - mu[:, j].reshape([-1, 1])			# dim x 1
			diff_t = np.transpose(diff) 
			weighted_diff += gamma_ * diff.dot(diff_t)

		sigma_list[j] = weighted_diff / den


	# Log-likelihood training data
	llik_new = utils.log_likelihood(n_states, time_steps, gamma, csi, data_train, A, pi, mu, sigma_list)
	llik_list.append(llik_new / time_steps)

	# Test data

	# E-step:
	
	# Alpha
	alpha_log = utils.alpha_log(data_test, pi, mu, sigma_list, A, n_states)
	
	# Beta
	beta_log = utils.beta_log(data_test, pi, mu, sigma_list, A, n_states)

	# Gamma
	gamma_log = utils.gamma_log(alpha_log, beta_log)

	gamma = np.exp(gamma_log)

	# Csi
	csi_log = np.zeros([time_steps-1, n_states, n_states])
	for t in range(time_steps-1):
		aux = np.zeros([n_states, n_states])
		for m in range(n_states):
			for l in range(n_states):
				aux[m, l] = alpha_log[m, t] + beta_log[l, t+1] + np.log(A[l, m]) + np.log(utils.multivariate_gaussian(data_test[t+1, :], mu[:, l], sigma_list[l]))	
		b = np.max(aux)
		den = b + np.log(np.sum(np.exp(aux - b)))
		
		for i in range(n_states):
			for j in range(n_states):
				csi_log[t, j, i] = alpha_log[i, t] + beta_log[j, t+1] + np.log(A[j, i]) + np.log(utils.multivariate_gaussian(data_test[t+1, :], mu[:, j], sigma_list[j])) - den

	csi = np.exp(csi_log)

	# Log-likelihood test data
	llik_test = utils.log_likelihood(n_states, time_steps, gamma, csi, data_test, A, pi, mu, sigma_list)
	llik_list_test.append(llik_test / time_steps)
	
	counter += 1

print('pi: \n', pi)
print('mu[0]: \n', mu[:, 0])
print('mu[1]: \n', mu[:, 1])
print('mu[2]: \n', mu[:, 2])
print('mu[3]: \n', mu[:, 3])
print('cov[0]: \n', sigma_list[0])
print('cov[1]: \n', sigma_list[1])
print('cov[2]: \n', sigma_list[2])
print('cov[3]: \n', sigma_list[3])
print('Trasition matrix: \n', A)

print('Normalized log-likelihood on training data:', llik_list[-1])
print('Normalized log-likelihood on test data:', llik_list_test[-1])
	
plt.plot(llik_list, 'c')
plt.title('Normalized log-likelihood - Training data')
plt.ylabel('Normalized log-likelihood')
plt.xlabel('Iterations')
if (save_plots):
	name = './Figures/llik_train.png'
	plt.savefig(name)
plt.show()
plt.clf()

plt.plot(llik_list_test, 'c')
plt.title('Normalized log-likelihood - Test data')
plt.ylabel('Normalized log-likelihood')
plt.xlabel('Iterations')
if (save_plots):
	name = './Figures/llik_test.png'
	plt.savefig(name)
plt.show()
plt.clf()


# Question 8
viterbi = np.zeros([n_states, time_steps])
best_path = np.zeros([time_steps])
pointer = np.zeros([n_states, time_steps])
for k in range(n_states):
	viterbi[k, 0] = np.log(pi[k] + 1e-12) + np.log(utils.multivariate_gaussian(data_train[0, :], mu[:, k], sigma_list[k]))

for t in range(1, time_steps):
	for i in range(n_states):
		viterbi_ = np.zeros([n_states, 1])
		for j in range(n_states):		
			viterbi_[j] = viterbi[j, t-1] + np.log(A[i, j])
		viterbi[i, t] = np.max(viterbi_ + np.log(utils.multivariate_gaussian(data_train[t, :], mu[:, i], sigma_list[i])))
		pointer[i, t] = np.argmax(viterbi_)

best_path[-1] = np.argmax(viterbi[:, -1])

for t in range(1, time_steps):
	idx = time_steps - t - 1
	best_path[idx] = pointer[int(best_path[idx+1]), idx+1]

plt.plot(best_path[0:100])
plt.xlabel('Time steps')
plt.ylabel('Most likely state')
plt.title('Viterbi alg. - Most likely sequence of states - Training data')
if (save_plots):
	name = './Figures/likely_states_viterbi_train.png'
	plt.savefig(name)
plt.show()
plt.clf()

colors = ['c', 'lightskyblue', 'mediumpurple', 'hotpink']

for m in range(n_states):
	color = colors[m]
	cluster_samples = data_train[np.where(best_path == m)[0]]

	plt.plot(cluster_samples[:, 0], cluster_samples[:, 1], 'o', c = color, label = 'Cluster'+' '+str(m))
	plt.scatter(mu[0, m], mu[1, m], marker='x', s = 100, c = k, linewidths = 5, zorder = 10)

	ellipse_data = utils.plot_ellipse(x_cent = mu[0, m], y_cent = mu[1, m], cov = sigma_list[m], mass_level = 0.9)
	plt.plot(ellipse_data[0], ellipse_data[1], c = color)

plt.legend(loc = 'upper left', scatterpoints=1)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.title('Viterbi alg. - Most likely states - Training data')

if (save_plots):
	name = './Figures/viterbi_train.png'
	plt.savefig(name)
plt.show()
plt.clf() 


# Question 9
# Alpha
alpha_log = utils.alpha_log(data_test, pi, mu, sigma_list, A, n_states)

# Beta
beta_log = utils.beta_log(data_test, pi, mu, sigma_list, A, n_states)

# Gamma
gamma_log = utils.gamma_log(alpha_log, beta_log)

gamma = np.exp(gamma_log)

plt.subplot(411)
plt.plot(gamma[0, 0:100], 'c')
plt.title("$p(z_t|x_1, \ldots , x_T)$ - HMM (EM parameters) - data")
plt.ylabel('State 0')

plt.subplot(412)
plt.plot(gamma[1, 0:100], 'lightskyblue')
plt.ylabel('State 1')

plt.subplot(413)
plt.plot(gamma[2, 0:100], 'mediumpurple')
plt.ylabel('State 2')

plt.subplot(414)
plt.plot(gamma[3, 0:100], 'hotpink')
plt.ylabel('State 3')
plt.xlabel('Time steps')

if (save_plots):
	name = './Figures/prob_zx_test.png'
	plt.savefig(name)

plt.show()
plt.clf()

# Question 10
likely_state_test = np.argmax(gamma, 0)
plt.plot(likely_state_test[0:100])
plt.ylabel('Most likely state')
plt.xlabel('Time steps')
plt.title('Alpha-Beta recursion - Most likely states - Test data')
if (save_plots):
	name = './Figures/likely_states_test.png'
	plt.savefig(name)
plt.show()
plt.clf()


# Question 11
viterbi = np.zeros([n_states, time_steps])
best_path = np.zeros([time_steps])
pointer = np.zeros([n_states, time_steps])
for k in range(n_states):
	viterbi[k, 0] = np.log(pi[k] + 1e-12) + np.log(utils.multivariate_gaussian(data_test[0, :], mu[:, k], sigma_list[k]))

for t in range(1, time_steps):
	for i in range(n_states):
		viterbi_ = np.zeros([n_states, 1])
		for j in range(n_states):		
			viterbi_[j] = viterbi[j, t-1] + np.log(A[i, j])
		viterbi[i, t] = np.max(viterbi_ + np.log(utils.multivariate_gaussian(data_test[t, :], mu[:, i], sigma_list[i])))
		pointer[i, t] = np.argmax(viterbi_)

best_path[-1] = np.argmax(viterbi[:, -1])

for t in range(1, time_steps):
	idx = time_steps - t - 1
	best_path[idx] = pointer[int(best_path[idx+1]), idx+1]

plt.plot(best_path[0:100])
plt.ylabel('Most likely state')
plt.xlabel('Time steps')
plt.title('Viterbi alg. - Most likely states - Test data')
if (save_plots):
	name = './Figures/likely_states_viterbi_test.png'
	plt.savefig(name)
plt.show()
plt.clf()

#print(best_path==likely_state_test)

