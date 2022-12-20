import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T

def get_cifar10(root, size):
    transform = T.Compose([T.Resize((size, size)),
                                  T.ToTensor(),
                                  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
