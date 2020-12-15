import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image as I

from efficientnet_pytorch import EfficientNet

#variabili che indicano il nome delle due colonne del dataset
c_filepath = "filepath"
c_cracks = "cracks"

class ConcreteDataset(Dataset):
    def __init__(self, df, transforms=None, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
	#row = self.df.iloc[idx]
        img = I.open(self.df.iloc[idx][c_filepath])
        
        if self.transforms:
            #when using albumation we have to pass data as dictionary to transforms.
            #The output after transformation is also a dictionary
            #we need to take the value from dictionary. That is why we are giving an image at the end.
            img = self.transforms(**{"image": np.array(img)})["image"]
        
        if self.is_test:
            return img
        else:
            cracks_tensor = torch.tensor([self.df.iloc[idx][c_cracks]], dtype=torch.float32)
            return img, cracks_tensor

class ConcreteModel(nn.Module):
    def __init__(self, eff_name = 'efficientnet-b0', n_outs = 1):
        super().__init__()
        #downloading the pretrained efficientnet model
        self.backbone = EfficientNet.from_pretrained(eff_name)
        #getting the number of input layers in the classifiers
        in_features = getattr(self.backbone,'_fc').in_features
        #replacing the existing classifier with new one
        self.classifier = nn.Sequential(nn.Linear(in_features, in_features//2),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(in_features//2, in_features//4),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(in_features//4, n_outs))
    
    def forward(self, input_of_model):
        
        """
        here the input shape is torch.Size([64, 3, 227, 227])
        we need to extract the features. In my understanding it means taking the output before passing to classifier
        https://github.com/lukemelas/EfficientNet-PyTorch#example-feature-extraction
        """
        out_before_classifier = self.backbone.extract_features(input_of_model) #the output size is torch.Size([64, 1280, 7, 7])
        
        #to convert the 7x7 to 1x1 we use a adaptive average pool 2d
        pool_output = F.adaptive_avg_pool2d(out_before_classifier, 1) #the output is torch.Size([64, 1280, 1, 1])
        
        """
        now before passing to the classifier, we need to flatten it. Using view operation for the same
        the size parameter is the length on a particular axis. size(0) = 64 size(1) = 1280 size(2) and size(3) is 1
        """
        classifier_in = pool_output.view(pool_output.size(0),-1) #this operation will convert torch.Size([64, 1280, 1, 1]) to torch.Size([64, 1280])
        
        #this is then fed into a custom classifier which outputs the predicition
        classifier_out = self.classifier(classifier_in) #the classifier output will be of size torch.Size([64, 1])
        return classifier_out