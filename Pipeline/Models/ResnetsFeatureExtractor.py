import torch
import torch.nn as nn

class Resnet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        else:
            self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights = None, num_classes=5)
        self.classify = False
        self.in_features = self.resnet18.fc.in_features # 512
        self.resnet18.decoder = nn.ConvTranspose2d(self.in_features, out_channels=3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        if (self.classify):
            out = self.resnet18(x)
        else:
            identity = x
            out = self.resnet18.conv1(x)
            out = self.resnet18.bn1(out)
            out = self.resnet18.relu(out)
            out = self.resnet18.maxpool(out)
            out = self.resnet18.layer1(out)
            out = self.resnet18.layer2(out)
            out = self.resnet18.layer3(out)
            out = self.resnet18.layer4(out)
            out = self.resnet18.decoder(out)
            out += identity
        return out
    
    def get_model(self):
        return self.resnet18
    
    def get_type(self):
        return "Resnet18"
    
class Resnet34FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights='ResNet34_Weights.DEFAULT')
        else:
            self.resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights = None, num_classes=5)
        self.classify = False
        self.in_features = self.resnet34.fc.in_features # 1024
        self.resnet34.decoder = nn.ConvTranspose2d(self.in_features, out_channels=3, kernel_size=32, stride=32, padding=0)   

    def forward(self, x):
        if (self.classify):
            out = self.resnet34(x)
        else:
            identity = x
            out = self.resnet34.conv1(x)
            out = self.resnet34.bn1(out)
            out = self.resnet34.relu(out)
            out = self.resnet34.maxpool(out)
            out = self.resnet34.layer1(out)
            out = self.resnet34.layer2(out)
            out = self.resnet34.layer3(out)
            out = self.resnet34.layer4(out)
            out = self.resnet34.decoder(out)
            out += identity
        return out

    def get_model(self):
        return self.resnet34
    
    def get_type(self):
        return "Resnet34"
    
class Resnet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
        else:
            self.resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights = None, num_classes=100)
        self.classify = False
        self.in_features = self.resnet50.fc.in_features # 2048
        self.resnet50.decoder = nn.ConvTranspose2d(self.in_features, out_channels=3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        if (self.classify):
            out = self.resnet50(x)
        else:
            identity = x
            out = self.resnet50.conv1(x)
            out = self.resnet50.bn1(out)
            out = self.resnet50.relu(out)
            out = self.resnet50.maxpool(out)
            out = self.resnet50.layer1(out)
            out = self.resnet50.layer2(out)
            out = self.resnet50.layer3(out)
            out = self.resnet50.layer4(out)
            out = self.resnet50.decoder(out)
            out += identity
        return out

    # Instead of passing the identity here, can we do it in the other function?
    def get_model(self):
        return self.resnet50
    
    def get_type(self):
        return "Resnet50"
