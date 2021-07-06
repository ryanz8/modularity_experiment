import torch
import torchvision
import torchvision.transforms as transforms

def _MNIST(transform):
    data = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(data, [50000, 10000])

    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    return train_set, val_set, test_set

def MNIST():
    input_size = (1, 28, 28)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

    return _MNIST(transform) + (input_size,)

def load_data(dataset_name, params):
    batch_size = params['batch_size']

    if dataset_name == 'MNIST':
        train_set, val_set, test_set, input_size = MNIST()
    else:
        raise(Exception("Unknown dataset name: " + dataset_name))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False, num_workers=6, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, input_size