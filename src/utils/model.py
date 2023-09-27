import torch.nn as nn

from torchvision.models.resnet import resnet18


class ConcreteModel(nn.Module):
    def __init__(self, n_classes=2, verbose=True):
        super().__init__()
        self.classes = n_classes

        # Get resnet pretrained model
        self.model = resnet18(weights="IMAGENET1K_V1")

        # Get the number of input layers in the classifier
        in_features = self.model.fc.in_features
        in_features_lv2 = in_features // 2
        in_features_lv3 = in_features // 6

        # Replacing the existing classifier with new one
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, in_features_lv2),
            nn.Dropout(p=0.4),
            nn.Linear(in_features_lv2, in_features_lv3),
            nn.Dropout(p=0.3),
            nn.Linear(in_features_lv3, n_classes)
        )
        self.verbose = verbose
        if self.verbose:
            print("----- ConcreteModel ----")
            print("New Classifier Stats: ")
            print("Input features 1° Linear: ", str(in_features))
            print("Input features 2° Linear: ", str(in_features_lv2))
            print("Input features 3° Linear: ", str(in_features_lv3))
            print("Output 3° Linear: ", str(self.get_output_classes()))
            print("------------------------")

    def get_output_classes(self):
        return self.classes

    def forward(self, input_data):
        return self.model(input_data)


def get_model(device: str = None, fine_tune: bool = False, n_classes: int = 2, verbose: bool = True):
    nn_model = ConcreteModel(n_classes=n_classes, verbose=verbose)
    if device:
        nn_model.to(device)

    # If fine_tune then we only train the final classifier, otherwise we train also the backbone
    if fine_tune:
        for parameter in nn_model.parameters():
            parameter.requires_grad = False
        for parameter in nn_model.model.fc.parameters():
            parameter.requires_grad = True

    return nn_model
