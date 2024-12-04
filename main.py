import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from models_to_test import *
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
#provare ad aggiungere un layer convolutivo come projector per il fine tuning.

#Common parameters that we want to keep the same for all models
batch_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 30
criterion = nn.CrossEntropyLoss()
patience = 10
early_stopping = True
# to use the GPU
print(torch.cuda.is_available()) #Check if cuda is_available

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2,
                                          drop_last=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


augmentations = {
    'basic': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'flip': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'color_jitter': transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'rotation': transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'random_crop_flips': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ]),
    'all': transforms.Compose([
        transforms.RandomCrop(32, padding=4),     
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add more augmentations as needed
}





#Get the optimizer(SGD or Adam)
def get_optimizer(model, optimizer_type):
    if optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=0.001)
    # Add more optimizers as needed
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

#Get the scheduler(MultiStepLR or CosineAnnealingLR)
def get_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == 'MultiStepLR':
        return MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, **kwargs)
    # Add more schedulers as needed
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# define train and test function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        losses.append(loss.item())
    return np.mean(losses)


def test(model, device, test_loader, val=False):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # Make sure criterion is defined here as well

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate the average loss over all batches
    test_loss /= len(test_loader)  # Divide by number of batches

    mode = "Val" if val else "Test"
    print('\{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        mode,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = correct / len(test_loader.dataset)
    return test_loss, test_acc


# Main loop:

def train_and_evaluate(model, device, trainloader, valloader, optimizer_type, scheduler_config):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0
    no_acc_increment = 0
    optimizer = get_optimizer(model, optimizer_type)
    scheduler = get_scheduler(optimizer, **scheduler_config) if scheduler_config else None
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, trainloader, optimizer, epoch)  # Assuming `train` function is defined
        train_losses.append(train_loss)
        
        val_loss, val_acc = test(model, device, valloader, val=True)  # Assuming `test` function is defined
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if scheduler:
            scheduler.step()
        # Update best accuracy and early stopping counter(if early_stopping is True)
        if(early_stopping):
            if val_acc > best_acc:
                no_acc_increment = 0
                best_acc = val_acc
            else:
                no_acc_increment += 1
            
            if no_acc_increment == patience:
                print(f"Model early stopped at epoch {epoch}")
                break

    return train_losses, val_losses, val_accuracies

def makeplots(all_train_losses, all_val_losses, all_val_accuracies, model_names, filename):
    # Training Loss Plot
    for i, train_losses in enumerate(all_train_losses):
        plt.plot(train_losses, label=f'{model_names[i]}', color=f"C{i}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.savefig(f"./plots/{filename}_train.png")
    plt.clf()

    # Validation Loss Plot
    for i, val_losses in enumerate(all_val_losses):
        plt.plot(val_losses, label=f'{model_names[i]}', color=f"C{i}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(f"./plots/{filename}_val.png")
    plt.clf()

    # Validation Accuracy Plot
    for i, val_accuracies in enumerate(all_val_accuracies):
        plt.plot(val_accuracies, label=f'{model_names[i]}', color=f"C{i}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(f"./plots/{filename}_acc.png")
    plt.clf()

# Main function to train, evaluate, and plot multiple models with different hyperparameters
def main(models_config, device):
    all_train_losses, all_val_losses, all_val_accuracies = [], [], []
    model_names = []
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True)
    for config in models_config:
        model = config['model']
        optimizer_type = config['optimizer_type']
        scheduler_config = config['scheduler_config']

        # Select augmentation based on the model's specific configuration
        augmentation_type = config.get('augmentation')
        train_transform = augmentations[augmentation_type]
        
        
        # create a split for train/validation. We can use early stop
        trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
        trainset.dataset.transform = train_transform
        valset.dataset.transform = augmentations['basic']
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2,
                                                  drop_last=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2,
                                                  drop_last=False)
        print(f"Training Model: {config['name']} with optimizer: {optimizer_type},scheduler: {scheduler_config.get('scheduler_type')} on data augmented with: {augmentation_type}")
        train_losses, val_losses, val_accuracies = train_and_evaluate(
            model, device, trainloader, valloader, optimizer_type, scheduler_config
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_accuracies.append(val_accuracies)
        model_names.append(config['name'])

    # Plot comparison of all models
    makeplots(all_train_losses, all_val_losses, all_val_accuracies, model_names, "comparison_feature_extraction")
    for config in models_config:
        test_loss, test_acc = test(config['model'],device, testloader)
        print(f"Test accuracy of model {config['name']} : {test_acc}")
        print(f"Test loss of model {config['name']} : {test_loss}")




#Function to prepare different pretrained models, change classification layer with an MLP with correct dimensions
#for each model, and freeze all the layers of the nets except for the final MLP.

def change_classifier_and_freeze():

    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    mobilenetv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    resnet18.fc = nn.Sequential(nn.Linear(512, 256), 
                           nn.ReLU(),
                           nn.Dropout(p=0.5), 
                           nn.Linear(256, 10))

    classifier_layers = list(vgg16.classifier.children())[:-1]
    vgg16.classifier = nn.Sequential(*classifier_layers,
    nn.Linear(4096, 512, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 10, bias = True))

    classifier_layers = list(mobilenetv2.classifier.children())[:-1]
    mobilenetv2.classifier = nn.Sequential(*classifier_layers,
    nn.Linear(1280, 512, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 10, bias = True))

    classifier_layers = list(densenet121.classifier.children())[:-1]
    densenet121.classifier = nn.Sequential(*classifier_layers,
    nn.Linear(1024, 512, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 10, bias = True))
    #Freeze all the parameters of all the models
    models_to_freeze = [resnet18,vgg16,mobilenetv2,densenet121]
    for model in models_to_freeze:
        for param in model.parameters():
            param.requires_grad = False

    #Unfreeze the parameters of the classification layers


    for layer in resnet18.fc:
        for param in layer.parameters():
            param.requires_grad = True
    for model in [vgg16,mobilenetv2,densenet121]:
        for layer in model.classifier:
            for param in layer.parameters():
                param.requires_grad = True

    return resnet18,vgg16,mobilenetv2,densenet121
#Here we define models and run experiments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet18,vgg16,mobilenetv2,densenet_cifar = change_classifier_and_freeze()



resnet18.to(device)
vgg16.to(device)
mobilenetv2.to(device)
densenet_cifar.to(device)
# Define models with different configurations
models_config = [
    {
        'name': 'resnet18',
        'model': resnet18,
        'optimizer_type': 'Adam',
        'scheduler_config': {
        },
        'augmentation': 'random_crop_flips'
    },
    {
        'name': 'vgg16',
        'model': vgg16,
        'optimizer_type': 'Adam',
        'scheduler_config': {
        },
        'augmentation': 'random_crop_flips'
    },
    {
        'name': 'mobilenetv2',
        'model': mobilenetv2,
        'optimizer_type': 'Adam',
        'scheduler_config': {
        },
        'augmentation': 'random_crop_flips'
    },
    {
        'name': 'densenet_cifar',
        'model': densenet_cifar,
        'optimizer_type': 'Adam',
        'scheduler_config': {
        },
        'augmentation': 'random_crop_flips'
    }
]

# Run main function
main(models_config, device)
