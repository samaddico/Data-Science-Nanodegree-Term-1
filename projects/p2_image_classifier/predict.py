import argparse
import torch
from torchvision import models
import numpy as np
from PIL import Image


def load_checkpoint(filepath):
    
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else: checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.resnet101(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    model.idx_to_class = checkpoint['idx_to_class']
    model.cat_to_name = checkpoint['cat_to_name']
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open Image
    im = Image.open(image)

    # Scale Image with shortest side as 256
    width, height = im.size
    shortest = min(width, height)
    width = int(256 * width / shortest)
    height = int(256 * height / shortest)
    im.thumbnail((width, height))

    # Center Crop
    left = (width - 224) / 2
    upper = (height - 224) / 2
    right = (width + 224) / 2
    lower = (height + 224) / 2
    im = im.crop((left, upper, right, lower))

    # Image to Numpy Array
    np_image = np.array(im) / 255

    # Normalize Image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose Image & Returns Numpy Array
    np_image = np_image.transpose((2,0,1))
    return np_image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()

    # Processing the image
    image = process_image(image_path)
    img = torch.tensor(image).float()
    img = img.unsqueeze(0)

    with torch.no_grad():
        
        prediction = model.forward(img)

        # Calculating Probabilities
        probabilities = torch.exp(prediction)
        top_prob, top_idx = probabilities.topk(topk, dim = 1)
        
        # Turn into Numpy Array
        top_prob = top_prob.data.numpy().squeeze(0)
        top_idx = top_idx.data.numpy().squeeze(0)
        
        # Change from index to class
        top_class = [model.idx_to_class[each] for each in list(top_idx)]

    return top_prob, top_class
    

def print_cat(top_prob, top_class):
    print()
    for idx, prob in np.nditer([top_class, top_prob]):
        cat = model.cat_to_name.get(str(idx))
        print(f'Probability: {prob*100:.1F}%  Class: {idx}  Flower: {cat}')
    print()


parser = argparse.ArgumentParser(description='Process flower images.')
parser.add_argument('checkpoint', help="Returns the top flower name and class probability.")
parser.add_argument('--top_k', type=int, default=1, help="Returns the top K likely flower names with their probability.")
parser.add_argument('--gpu', action="store_true", help="Utilizes gpu instead of cpu.")

args = parser.parse_args()

image_path = args.checkpoint
k = args.top_k

device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
model = load_checkpoint('checkpoint.pth')

top_prob, top_cat = predict(image_path, model, k)
print_cat(top_prob, top_cat)
