import torch
import numpy as np

def evaluate(model, input_data, input_dimension):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	results = np.zeros(input_dimension)
	model.eval()
	with torch.no_grad():
		for i, (xb) in enumerate(input_data):
			xb = xb.to(device)
			bs = xb.shape[0]
			results[i*bs : i*bs+bs] = torch.max(model(xb), 1).indices.cpu().numpy()
	return results