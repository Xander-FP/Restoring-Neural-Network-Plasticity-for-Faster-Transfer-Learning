import torch
import torch.nn as nn

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load full pretrained MobileNetV2
        if pretrained:
            self.mobileNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')
        else:
            self.mobileNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', num_classes=5)
        self.classify = False
        self.in_features = self.mobileNet.classifier[1].in_features
        self.mobileNet.decoder = nn.ConvTranspose2d(self.in_features, out_channels=3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        if (self.classify):
            out = self.mobileNet(x)
        else:
            out = self.mobileNet.features(x) 
            out = self.mobileNet.decoder(out) 
        return out
    
    def get_model(self):
        return self.mobileNet
    
    def get_type(self):
        return "MobileNet"
