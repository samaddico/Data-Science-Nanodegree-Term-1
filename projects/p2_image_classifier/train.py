import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    return trainloader, validloader, testloader

def load_model(arch, learning_rate, hidden_units):
    
    # Loads three differnt Resnet Models
    if arch == "densenet121":
        model = models.densenet121(pretrained = True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained = True)
    else:
        model = models.resnet101(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    
    # Initializes classifier based on user input of hidden units
    if arch == "densenet121":
        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))    
    elif arch == "vgg16":
        model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))        
    else:
        model.fc = nn.Sequential(nn.Linear(2048, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

    # Loads other presaved data
    model.criterion = nn.NLLLoss()
    if arch == "resnet101":
        model.optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)
    else:
        model.optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    print()
    print("Successfully loaded {} with {} hidden units and a learning rate of {}.".format(arch, hidden_units, learning_rate))
    print()
    
    return model

def train(trainloader, validloader, epochs):
    
    batch_print = 25
    model.to(device)
    
    for e in range(epochs):
        # Initialize batch and training loss to 0 for epochs
        batch = 0
        training_loss = 0

        for images, labels in trainloader:
            # Batch # out of 103 batches
            batch += 1
            # Sending images to gpu
            images, labels = images.to(device), labels.to(device)
            # Output and Loss
            output = model.forward(images)
            loss = model.criterion(output, labels)
            training_loss += loss.item()
            #Autograde
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # Print every 25 batches
            if batch % batch_print == 0:
                # Initializing validation loss and accuracy
                model.eval()
                accuracy = 0
                valid_loss = 0
                with torch.no_grad():
                    for images, labels in validloader:
                        # Sending images to gpu
                        images, labels = images.to(device), labels.to(device)
                        # Output and Loss
                        prediction = model.forward(images)
                        valid_loss += model.criterion(prediction, labels).item()
                        # Calculating accuracy
                        probabilities = torch.exp(prediction)
                        top_prob, top_class = probabilities.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Printing results every 25 batches
                print(f'Epoch {e+1} Batch {batch} - '
                      f'Training Loss: {training_loss/batch:.3F}, '
                      f'Validate Loss: {valid_loss/len(validloader):.3F}, '
                      f'Validate Accuracy: {accuracy*100/len(validloader):.1F}%')
                model.train()

    print("Training Complete!")
    print()
    
def save_model(save_dir):
    if arch == "resent101":
        classifer = model.fc
        state_dict = model.fc.state_dict()
    else:
        classifier = model.classifier
        state_dict = model.fc.state_dict()
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    checkpoint = {'arch': arch,
                  'epochs': epochs,
                  'hidden_units': hidden_units,
                  'classifier': classifier,
                  'state_dict': state_dict,
                  'optimizer': model.optimizer,
                  'idx_to_class': idx_to_class,
                 }
    torch.save(checkpoint, save_dir + 'checkpoint.pth')
                
parser = argparse.ArgumentParser(description='Trains model using a directory of flowers.')
parser.add_argument('train', help="Input directory for training/validating data.")
parser.add_argument('--save_dir', help="Input save directory.")
parser.add_argument('--arch', choices=['vgg16', 'resnet101', 'densenet121'], default='resnet101', help="Choose between either vgg-16, resnet-101, or densenet-121 models to train on - default resnet101.")
parser.add_argument('--learning_rate', type=float, default=0.03, help="Enter the learning rate - default 0.03.")
parser.add_argument('--hidden_units', type=int, default=512, help="Enter the number of hiddlen units - default 512.")
parser.add_argument('--epochs', type=int, default=1, help="Enter number of additional epochs.")
parser.add_argument('--gpu', action="store_true", help="Utilizes gpu instead of cpu.")

args = parser.parse_args()
data_dir = args.train
save_dir = args.save_dir
arch = args.arch
epochs = args.epochs
hidden_units = args.hidden_units
learning_rate = args.learning_rate

device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
model = load_model(arch, learning_rate, hidden_units)

trainloader, validloader, testloader = load_data(data_dir)
train(trainloader, validloader, epochs)
if save_dir != None:
    save_model(save_dir)