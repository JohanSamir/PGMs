import numpy as np
from mean_field import mean_field
from gibbs_sampling import gibbs_sampling
import utils

import argparse

parser = argparse.ArgumentParser(description='IFT6269 HW5')
parser.add_argument('--init', type = int, default = 1, help = 'MF initialization: 1, 2, 3, 4')
args = parser.parse_args()


#var, mu_hat = gibbs_sampling()
#print('Empirical SD of GS MC estimates of mu: \n', var)
#print('Empirical mean of GS MC estimates of mu: \n', mu_hat)

tau_hat = mean_field(args.init)
print('Mean field estimate of mu: \n', tau_hat)

#dist = utils.dist(tau_hat, mu_hat)
#print('L1 distance between GS and MF estimates of mu: \n', dist)

