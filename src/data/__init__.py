

from .mnist import MNIST
from .cifar10 import get_cifar10
import os 
def get_data(flags):
    name = flags.data
    if name== "mnist":
        return MNIST(flags)

    elif name == 'cifar10':
        # train dataset / valid dataset /  number of classes / input channel too
        return *get_cifar10(os.path.join(flags.data_root, 'cifar10'), flags.img_size), 10, 3   