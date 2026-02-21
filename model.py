import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet50(pretrained=True)
    
    # Change final layer to 50 outputs
    model.fc = nn.Linear(model.fc.in_features, 49)
    
    return model