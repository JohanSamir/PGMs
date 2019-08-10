import numpy as np
import argparse

from question1 import question1
from question2 import question2
from question3 import question3
from question4 import question4
from question5 import question5


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='IFT6269 HW2')
	parser.add_argument('--question', type = int, default = 1, help = 'Question number: 1, 2, 3, 4, 5 ')
	parser.add_argument('--all-questions', action = 'store_true', default = False, help = 'True for all questions')
	parser.add_argument('--save-plots', action = 'store_true', default = True, help = 'Disable figures saving')
	args = parser.parse_args()


	
	if (args.all_questions or (args.question == 1)):	

		# Question 1
		_ = question1('A', 'train', args.save_plots)
		_ = question1('B', 'train', args.save_plots)
		_ = question1('C', 'train', args.save_plots)
	
	elif (args.all_questions or (args.question == 2)):

		# Question 2
		_ = question2('A', 'train', args.save_plots)
		_ = question2('B', 'train', args.save_plots)
		_ = question2('C', 'train', args.save_plots)
	

	elif (args.all_questions or (args.question == 3)):
		
	# Question 3
		_ = question3('A', 'train', args.save_plots)
		_ = question3('B', 'train', args.save_plots)
		_ = question3('C', 'train', args.save_plots)

	elif (args.all_questions or (args.question == 4)):

		# Question 4
		question4('A', 'train')
		question4('B', 'train')
		question4('C', 'train')
		question4('A', 'test')
		question4('B', 'test')
		question4('C', 'test')

	
	elif (args.all_questions or (args.question == 5)):
		
		# Question 5
		question5('A', 'train', args.save_plots)
		question5('B', 'train', args.save_plots)
		question5('C', 'train', args.save_plots)

		question5('A', 'test', args.save_plots, no_outputs = True)		
		question5('B', 'test', args.save_plots, no_outputs = True)
		question5('C', 'test', args.save_plots, no_outputs = True)

