import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

def getModelInputSize():
    #width, length, channels
    return (227, 227, 3)

class ConcreteModel(nn.Module):
    def __init__(self, eff_name = 'efficientnet-b0', n_outs = 1):
        super().__init__()
        self.classes = n_outs
        #downloading the pretrained efficientnet model
        self.backbone = EfficientNet.from_pretrained(eff_name)
        #getting the number of input layers in the classifiers
        in_features = getattr(self.backbone, '_fc').in_features
        #replacing the existing classifier with new one
        print("----- ConcreteModel ----")
        print("New Classifier Stats: ")
        print("Input features 1° Linear: ", str(in_features))
        print("Input features 2° Linear: ", str(in_features//2))
        print("Input features 3° Linear: ", str(in_features//4))
        print("Output 3° Linear: ", str(n_outs))
        print("------------------------")
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            #Dropout azzera il valore di alcuni neuroni con prob bernoulliana con p definito
            #è provato che migliora il rendimento della rete
            nn.Dropout(p=0.2),
            nn.Linear(in_features//2, in_features//4),
            nn.Dropout(p=0.2),
            nn.Linear(in_features//4, n_outs)
        )
        
    def n_output_classes(self):
        return self.classes
    
    def forward(self, input_of_model):
        """
        here the input shape is torch.Size([64, 3, 227, 227]), we need to extract the features.
        https://github.com/lukemelas/EfficientNet-PyTorch#example-feature-extraction
        """
        out_before_classifier = self.backbone.extract_features(input_of_model)
        #the output size is torch.Size([64, 1280, 7, 7])
        
        #to convert the 7x7 to 1x1 we use a adaptive average pool 2d
        pool_output = F.adaptive_avg_pool2d(out_before_classifier, 1)
        #the output is torch.Size([64, 1280, 1, 1])
        
        """
        now before passing to the classifier, we need to flatten it. Using view operation for the same
        the size parameter is the length on a particular axis. size(0) = 64 size(1) = 1280 size(2) and size(3) is 1
        """
        classifier_in = pool_output.view(pool_output.size(0), -1)
        #this operation will convert torch.Size([64, 1280, 1, 1]) to torch.Size([64, 1280])
        
        #this is then fed into a custom classifier which outputs the predicition
        classifier_out = self.classifier(classifier_in)
        #the classifier output will be of size torch.Size([64, n_outs])
        return classifier_out