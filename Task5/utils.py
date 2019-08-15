import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt



def sigmoid(x):
	
	sig = 1.0/ (1.0 + np.exp(-x) )

	return sig 

def dist(tau_new, tau_old):
	
	dist = np.mean(np.abs(tau_new - tau_old))

	return dist

 
