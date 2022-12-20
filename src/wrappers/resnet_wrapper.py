

import torch 

class ResnetWrapper():
    def __init__(self, resnet, gap_module):
        self.resnet = resnet
        self.gap_module = gap_module
            
    def forward_resnet(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for m in self.resnet.layer1:
            x, out1 = forward_bottleneck(m, x)
        for m in self.resnet.layer2:
            x, out2 = forward_bottleneck(m, x)
        for m in self.resnet.layer3:
            x, out3 = forward_bottleneck(m, x)
        for m in self.resnet.layer4:
            x, out4 = forward_bottleneck(m, x)
        gap = self.gap_module((out1, out2, out3, out4))
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x + gap 
        x = self.resnet.fc(x)
        
        return x

def forward_bottleneck(model, x):
    assert model.__class__.__name__ == "Bottleneck"
    identity = x

    out = model.conv1(x)
    out = model.bn1(out)
    out = model.relu(out)
    out1 = out 
    
    out = model.conv2(out)
    out = model.bn2(out)
    out = model.relu(out)
    out2 = out 
    
    out = model.conv3(out)
    out = model.bn3(out)
    out3 = out
    if model.downsample is not None:
        identity = model.downsample(x)
    out += identity
    out = model.relu(out)

    return out, (out1, out2, out3)
