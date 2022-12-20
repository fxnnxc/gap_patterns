import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 
def get_cifar10(root, size):
    transform = T.Compose([T.Resize((size, size)),
                                  T.ToTensor(),
                                  T.Normalize(mean, std)
                                 ])
    train_ds = datasets.CIFAR10(root = root,  
                                    train = True,  
                                    transform = transform,
                                    download=True)
    valid_ds = datasets.CIFAR10(root = root,
                                    train = False, 
                                    transform =transform,
                                    download=True)
    return train_ds, valid_ds
