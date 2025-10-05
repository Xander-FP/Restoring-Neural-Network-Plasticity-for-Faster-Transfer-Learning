import torch
import torch.nn as nn
import torchvision

class ViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.vit = torchvision.models.vision_transformer.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        else:
            self.vit = torchvision.models.vision_transformer.vit_b_16(weights=None)
        self.classify = False
        self.in_features = self.vit.heads.head.in_features
        # self.vit.decoder = nn.ConvTranspose2d(self.in_features, out_channels=3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        if (self.classify):
            out = self.vit(x)
        # else:
            # out = self.vit.features(x)  
            # out = self.vit.decoder(out)
        return out
    
    def get_model(self):
        return self.vit
    
    def get_type(self):
        return "VIT_B_16"
