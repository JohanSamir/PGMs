import numpy as np
import utils
from question1 import question1
from question2 import question2
from question3 import question3

def question4(dataset_id, train_test):

	error_model1 = question1(dataset_id, train_test, no_outputs = True)
	error_model2 = question2(dataset_id, train_test, no_outputs = True)
	error_model3 = question3(dataset_id, train_test, no_outputs = True)

	print('Misclassification error for model 1 on dataset',dataset_id,train_test,':',error_model1)
	print('Misclassification error for model 2 on dataset',dataset_id,train_test,':',error_model2)
	print('Misclassification error for model 3 on dataset',dataset_id,train_test,':',error_model3)
 
