import argparse
import os
import json

from model import ShuffleNet

from torchvision import transforms
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np

def get_transformer():
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])

  transformer = transforms.Compose([
      transforms.Resize(128),
      transforms.CenterCrop(128),
      transforms.ToTensor(),
      normalize
  ])
  return transformer

def preprocess(image, transformer):
  x = transformer(image)
  return Variable(x.unsqueeze(0))

def infer(args):
  # make ShuffleNet model
  print('Creating ShuffleNet model')
  net = ShuffleNet(num_classes=args.num_classes, in_channels=3)
  
  # load trained checkpoint
  print('Loading checkpoint')
  checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
  net.load_state_dict(checkpoint['state_dict'])

  print('Loading index-class map')
  with open(args.idx_to_class, 'r') as f:
      mapping = json.load(f)

  # image transformer
  transformer = get_transformer()

  # make input tensor
  print('Loading image')
  image = Image.open(args.image)
  print('Preprocessing')
  x = preprocess(image, transformer)

  # predict output
  print('Inferring on image {}'.format(args.image))
  y = net(x)
  top_idxs = np.argsort(y.data.cpu().numpy().ravel()).tolist()[-10:][::-1]
  print('==========================================')
  for i, idx in enumerate(top_idxs):
    key = str(idx)
    class_name = mapping[key][1]
    print('{}.\t{}'.format(i+1, class_name))
  print('==========================================')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('image', type=str, help='Path to image that we want to classify')
  parser.add_argument('checkpoint', type=str, help='Path to ShuffleNet checkpoint with trained weights')
  parser.add_argument('idx_to_class', type=str, help='Path to JSON file mapping indexes to class names')
  parser.add_argument('--num_classes', type=int, help='Number of classes to predict', default=1000)
  args = parser.parse_args()
  infer(args)
