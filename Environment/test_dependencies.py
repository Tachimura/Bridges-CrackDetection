"""
Quick script to test if any project dependencies are missed
"""
if __name__ == '__main__':
	try:
		import matplotlib
		import pandas
		import numpy
		import scipy
		from sklearn.metrics import accuracy_score
		from efficientnet_pytorch import EfficientNet
		import torch
		import albumentations
		from fastprogress.fastprogress import master_bar
		import torchvision
		print("All dependecies are satisfied!")
	except ModuleNotFoundError as e:
		print("Some dependencies missed:{}".format(e))