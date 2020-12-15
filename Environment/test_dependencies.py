"""
Quick script to test if any project dependencies are missed
"""
if __name__ == '__main__':
	try:
		import pandas
		import numpy
		import scipy
		import matplotlib
		import torch
		print("All dependecies are satisfied!")
	except ModuleNotFoundError as e:
		print("Some dependencies missed:{}".format(e))