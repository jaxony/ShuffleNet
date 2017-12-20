import argparse
import os

from model import ShuffleNet

from torchvision import transforms
import torch
from PIL import Image

def get_transformer(image):
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])

  transformer = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize
  ])
  return transformer

def preprocess(image, transformer):
  return transformer(image)

def infer(args):
  # make ShuffleNet model
  net = ShuffleNet(num_classes=1000, in_channels=3)
  
  # load trained checkpoint
  checkpoint = torch.load(args.checkpoint)
  net.load_state_dict(checkpoint['state_dict'])

  # image transformer
  transformer = get_transformer()

  # make input tensor
  image = Image.open(args.image)
  x = preprocess(image, transformer)

  # predict output
  y = net(x)
  print(y)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('image', help='Path to image that we want to classify')
  parser.add_argument('checkpoint', help='Path to ShuffleNet checkpoint with trained weights')
  parser.add_argument('idx_to_class', help='Path to JSON file mapping indexes to class names')
  args = parser.parse_args()
  infer(args)