


def get_encoder(flags):
    name = flags.encoder_name
    if name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        return model 
    elif name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        return model 
    elif name == 'resnet34':
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
        return model     
    elif name == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
        return model     
    elif name == 'resnet152':
        from torchvision.models import resnet152, ResNet152_Weights
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
        return model     
    else:
        raise ValueError(f"{flags.encoder_name} is not implemented")