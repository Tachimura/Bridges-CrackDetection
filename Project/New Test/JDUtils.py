# Import Generici
import numpy as np
import random

# Import per Pytorch
import torch
from torch.utils.data import Dataset, DataLoader

# Modifica immagini (colorazione, blur, ecc ecc)
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
# Gestione Immagini (Load, Save, Ecc...)
import cv2 as cv

#--------------------------------------
# Variabili che indicano il nome delle due colonne del ConcreteDatasetTraining
cd_filepath = "filepath"
cd_label = "label"
ConcreteDatasetColumns = [cd_filepath, cd_label]
#--------------------------------------

def init_random(deterministic_behaviour=False):
    # Seed per il random
    rseed = 6789
    if deterministic_behaviour:
        # Python random module.
        random.seed(rseed)
        # Numpy module.
        np.random.seed(rseed)
        # Imposto gli stessi seed per torch ed attivo solo roba deterministica
        torch.manual_seed(rseed)
        torch.cuda.manual_seed(rseed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(rseed)
        torch.manual_seed(rseed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Ritorno il seed usato per l'inizializzazione dei vari moduli
    return rseed

def getModelInputSize():
	#width, length, channels
	return (227, 227, 3)

# Ritorna il nome del modello di efficient-net che vogliamo usare
def getModelName():
	return "efficientnet-b3"
	#efficientnet-b0: 213 backbone params
	#efficientnet-b1: 301 backbone params
	#efficientnet-b2: 301 backbone params
	#efficientnet-b3: 340 backbone params
	# Come consiglio conviene al massimo usare b3 (poi dopo i parametri aumentano di moltissimo e l'accuracy sale di poco)
	# Giustamente se si ha una buona gpu si puÃ² anche salire fino a b8 che Ã¨ l'ultimo modello (attualmente: 17/12/2020)

# Questa classe permette di gestire il dataset di partenza, a cui perÃ² applico delle trasformazioni
# Se si vuole usare il dataset originario basta passare il dataframe di partenza.
class ConcreteDatasetTraining(Dataset):
	def __init__(self, df, transforms=None, is_test=False):
		self.df = df
		self.transforms = transforms
		self.is_test = is_test

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		img = cv.imread(str(self.df.iloc[idx][cd_filepath]))
		if self.transforms:
			img = self.transforms(**{"image": np.array(img)})["image"]

		if self.is_test:
			return img
		else:
			cracks_tensor = torch.tensor(self.df.iloc[idx][cd_label], dtype=torch.float32)
			return img, cracks_tensor

class ConcreteDatasetSegmented(Dataset):
	def __init__(self, ds, transforms=None):
		self.ds = ds
		self.transforms = transforms

	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		img = self.ds[idx]

		if type(img) == str:
			img = cv.imread(str(img))
		if self.transforms:
			img = self.transforms(**{"image": np.array(img)})["image"]

		return img, idx

def get_training_dataloader(data_dataframe, data_transformations = None, dl_bs = 64, dl_shuffle = False, num_dl_workers = 0):
	# Creo il ConcreteDataset
	data_cds = ConcreteDatasetTraining(df = data_dataframe, transforms = data_transformations)
	# Creo il DataLoader
	data_dl = DataLoader(dataset = data_cds, batch_size = dl_bs, shuffle = dl_shuffle, num_workers = num_dl_workers)
	return data_dl

def get_valid_dataloader(segmented_data_list, data_transformations = None, dl_bs = 64, num_dl_workers = 0):
	# Creo il ConcreteDatasetSegmented
	data_cds = ConcreteDatasetSegmented(ds = segmented_data_list, transforms = data_transformations)
	# Creo il DataLoader
	data_dl = DataLoader(dataset = data_cds, batch_size = dl_bs, shuffle = False, num_workers = num_dl_workers)
	return data_dl

#--------------------------------------
# Trasformazioni da applicare ai set di dati

# Trasformazioni per il set di training
def get_training_tfms(dataset_normalize_stats):
	size = getModelInputSize()
	return	A.Compose(
		[
			A.Resize(size[0], size[1]),
			A.RandomRotate90(p=0.5),
			A.Flip(p=0.5),
			A.ColorJitter(p=0.5),
			A.RandomGamma(p=0.5),
			A.RandomContrast(p=0.3),
			A.RandomBrightness(p=0.5),
			ToTensor(dataset_normalize_stats),
		])

# Trasformazioni x il set di validation
def get_validation_tfms(dataset_normalize_stats):
	size = getModelInputSize()
	return	A.Compose(
		[
			A.Resize(size[0], size[1]),
			ToTensor(dataset_normalize_stats),
		])
#--------------------------------------