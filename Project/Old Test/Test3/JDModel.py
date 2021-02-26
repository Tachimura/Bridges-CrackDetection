# Import per Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import per le reti di EfficientNet
from efficientnet_pytorch import EfficientNet

class ConcreteModel(nn.Module):
	def __init__(self, model_name='efficientnet-b0', n_classes=1):
		super().__init__()
		self.classes = n_classes

		# Downloading the pretrained efficientnet model
		self.backbone = EfficientNet.from_pretrained(model_name)

		# Getting the number of input layers in the classifiers
		in_features = self.backbone._bn1.num_features
		in_features_lv2 = in_features // 2
		in_features_lv3 = in_features // 6

		# Replacing the existing classifier with new one
		self.classifier = nn.Sequential(
			nn.Linear(in_features, in_features_lv2),
			#Dropout azzera il valore di alcuni neuroni con prob bernoulliana con p definito
			#è provato che migliora il rendimento della rete
			nn.Dropout(),#p=0.5),
			nn.Linear(in_features_lv2, in_features_lv3),
			nn.Dropout(),#p=0.3),
			nn.Linear(in_features_lv3, n_classes)
		)
		print("----- ConcreteModel ----")
		print("New Classifier Stats: ")
		print("Input features 1° Linear: ", str(in_features))
		print("Input features 2° Linear: ", str(in_features_lv2))
		print("Input features 3° Linear: ", str(in_features_lv3))
		print("Output 3° Linear: ", str(self.getOutputClasses()))
		print("------------------------")

	def getOutputClasses(self):
		return self.classes

	def forward(self, input):
		# nuova prova
		#print(input.shape)
		out_before_classifier = self.backbone.extract_features(input)
		#print(out_before_classifier.shape)
		pool_output = F.adaptive_avg_pool2d(out_before_classifier, 1)
		#print(pool_output.shape)
		classifier_in = pool_output.view(pool_output.size(0), -1)
		#print(classifier_in.shape)
		out = self.classifier(classifier_in)
		#print(out.shape)
		# All'output devo poi applicarci la softmax: out = nn.Softmax(dim=1)(out)
		return out

#--------------------------------------
# Metodi utili x l'uso della rete
def get_device():
	return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Metodo che ritorna una nuova copia del modello
def get_model(model_name = 'efficientnet-b0', lr = 1e-5, wd = 0.01, feature_extracting = False, opt_fn=torch.optim.Adam, device = None, n_classes = 1):
	device = device if device else get_device()
	model = ConcreteModel(model_name=model_name , n_classes=n_classes)

	#Se feature_training_from = 0, rialleno tutto il backbone, dunque non imposto a False i requires_grad dei layer
	if feature_extracting:
		for parameter in model.backbone.parameters():
			parameter.requires_grad = False

	opt = opt_fn(model.parameters(), lr=lr, weight_decay=wd)
	model = model.to(device)
	return model, opt

# Step per il training
def training_step(xb, yb, model, loss_fn, opt, device, scheduler):
	xb ,yb = xb.to(device), yb.to(device)
	opt.zero_grad()
	out = model(xb)
	loss = loss_fn(out, yb)
	loss.backward()
	opt.step()
	return loss.item()

# Step per la validazione
def validation_step(xb, yb, model, loss_fn, device):
	xb, yb = xb.to(device), yb.to(device)
	out = model(xb)
	loss = loss_fn(out, yb)
	return loss.item(), out

# Step per l'uso normale della rete
def test_step(xb, model, device):
	xb = xb.to(device)
	out = torch.max(nn.Softmax(dim=1)(model(xb)), 1).indices.cpu().numpy()
	# Ritorno le predizioni ed il numero di predizioni
	return out, xb.shape[0]
#--------------------------------------