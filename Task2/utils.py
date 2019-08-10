import numpy as np



def normalize(x):
	mean = np.mean(x, axis = 0)
	std = np.std(x, axis = 0)

	normalized = (x - mean)/(std + 1e-10)

	return normalized

def load_dataset(dataset_name = 'A', train_test = 'train', path = './hwk2data/'):
	
	filename = path + 'classification' + dataset_name + '_' + train_test + '.txt'

	all_data = []

	with open(filename) as file_:
		
		reader = file_.readlines()
		
		for row in reader:
			all_data.append([float(sample) for sample in row.split("\t")])

	
	all_data = np.asarray(all_data)

	x = all_data[:, 0:2]
	y = all_data[:, 2].reshape([-1, 1])

	return normalize(x), y 

def sigmoid(x):
	
	sig = 1.0/ (1.0 + np.exp(-x) )

	return sig 
