import torch
import torch.nn as nn


vgg_config = {
    'vgg-11': [32, 'P', 64, 'P', 128, 128, 'P', 256, 256, 'P', 256, 256, 'P']
}


class VGG(nn.Module):
    
    def __init__(self, spec, output_dim):
        super().__init__()
        self.features = self._feature_layer(vgg_config[spec])
        self.classifier = nn.Sequential(
            nn.Linear(256, output_dim)
        )  
        
    def forward(self, x):
    
        out = self.features(x)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        return out
      
    def _feature_layer(self, config):
        
        layers = []
        in_channels = 3
        
        for c in config:
            if c == 'P':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(c),
                           nn.ReLU(inplace=True)]
                in_channels = c
                
        return nn.Sequential(*layers)
