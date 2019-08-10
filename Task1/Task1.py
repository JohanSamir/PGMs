import numpy as np
import matplotlib.pyplot as plt


mean = 0.0
var = 1.0
num_samples = 5

# Item a
samples = np.random.normal(mean, var, num_samples)
print(samples)


# Item b
mean_hat = np.sum(samples) / num_samples
print('Mean estimator:', mean_hat)

var_hat = np.sum((samples - mean_hat)**2) / num_samples
print('Variance estimator:', var_hat)


# Item c
repeat_times = 10000
var_hat_repet = np.zeros(10000)

for i in range(0, repeat_times):
	
	samples = np.random.normal(mean, var, num_samples)
	mean_hat = np.sum(samples) / num_samples
	var_hat_repet[i] = np.sum((samples - mean_hat)**2) / num_samples

plt.hist(var_hat_repet, 50, color='violet')
plt.title('Histogram of 10000 estimates of Gaussian variance parameter')
plt.xlabel('Variance estimate')
plt.ylabel('Number of occurrencies')
plt.show()


# Item d
bias = np.mean(var_hat_repet) - var
print('Bias of variance estimator', bias)
variance = np.mean(var_hat_repet**2) - (np.mean(var_hat_repet))**2
print('Variance of variance estimator', variance)

# Item e
calc_bias = -var / num_samples
print('Calculated bias of variance estimator', calc_bias)
calc_variance = (2*(num_samples - 1) / num_samples**2) * var**4 
print('Calculated variance of variance estimator', calc_variance)






