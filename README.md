# ShuffleNet in PyTorch
An implementation of `ShuffleNet` in PyTorch. `ShuffleNet` is an efficient convolutional neural network architecture for mobile devices. According to the paper, it outperforms Google's MobileNet by a small percentage.

## What is ShuffleNet?
In one sentence, `ShuffleNet` is a ResNet-like model that uses residual blocks (called `ShuffleUnits`), with the main innovation being the use of pointwise, or 1x1, *group* convolutions as opposed to normal pointwise convolutions.

## Usage
Clone the repo:
```bash
git clone https://github.com/jaxony/ShuffleNet.git
```

Use the model defined in `model.py`:
```python
from model import *

# running on MNIST
net = ShuffleNet(num_classes=10, in_channels=1)
```

## Performance
The `ShuffleNet` implementation has been briefly tested on the MNIST dataset and achieves 90+% accuracy within the first 5 epochs, so the model can certainly learn. If anyone has the GPU resources to train `ShuffleNet` on ImageNet, do share the weights if you manage to train it successfully as in the original paper!

