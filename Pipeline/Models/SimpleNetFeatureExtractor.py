import torch
import torch.nn as nn

class SimpleNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.simpleNet = torch.hub.load('coderx7/simplenet_pytorch:v1.0.0', 'simplenetv1_5m_m1', pretrained=True)
        else:
            self.simpleNet = torch.hub.load('coderx7/simplenet_pytorch:v1.0.0', 'simplenetv1_5m_m1', num_classes=100)
        self.classify = False
        self.in_features = self.simpleNet.classifier.in_features
        self.simpleNet.decoder = nn.ConvTranspose2d(self.in_features, out_channels=3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        if (self.classify):
            out = self.simpleNet(x)
        else:
            out = self.simpleNet.features(x)  
            out = self.simpleNet.decoder(out)
        return out
    
    def get_model(self):
        return self.simpleNet
    
    def get_type(self):
        return "SimpleNet"
