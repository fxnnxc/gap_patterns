from .resnet_wrapper import ResnetWrapper
def get_wrapper(name, encoder, gap_module):
    if name== 'resnet':
        return ResnetWrapper(encoder, gap_module)