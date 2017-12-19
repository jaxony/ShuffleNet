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
from model import ShuffleNet

# running on MNIST
net = ShuffleNet(num_classes=10, in_channels=1)
```

## Performance

Trained on ImageNet (using the [PyTorch ImageNet example][imagenet]) with
`groups=3` and no channel multiplier. On the test set, got 62.2% top 1 and
84.2% top 5. Unfortunately, this isn't comparable to Table 5 of the paper,
because they don't run a network with these settings, but it is somewhere
between the network with `groups=3` and half the number of channels (42.8%
top 1) and the network with the same number of channels but `groups=8`
(32.4% top 1). The pretrained state dictionary can be found [here][tar], in
the [following
format](://github.com/pytorch/examples/blob/master/imagenet/main.py#L165-L171):

```
{
    'epoch': epoch + 1,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'best_prec1': best_prec1,
    'optimizer' : optimizer.state_dict()
}
```

[tar]: https://drive.google.com/file/d/12oGJsyDgp51LhQ7FOzKxF9nBsutLkE6V/view?usp=sharing
[imagenet]: https://github.com/pytorch/examples/tree/master/imagenet
