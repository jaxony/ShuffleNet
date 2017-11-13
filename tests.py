import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model import channel_shuffle, ShuffleNet, ShuffleUnit

# helper methods to make torch Variables based on shape
def make_variable(shape):
    return Variable(torch.FloatTensor(np.random.random(shape)))

def get_input(batchsize=1, num_channels=3, height=224, width=224):
    shape = (batchsize, num_channels, height, width)
    return make_variable(shape)

class TestShuffleUnit(unittest.TestCase):

    def test_stage3_concat(self):
        groups, in_channels, out_channels = 3, 240, 480
        x = get_input(num_channels=in_channels, height=28, width=28)

        unit = ShuffleUnit(
            in_channels,
            out_channels,
            groups=groups,
            grouped_conv=True,
            combine='concat'
            )
        out = unit.forward(x)

        self.assertEqual(0, np.any(out.data.size() != 
            (1, out_channels, 14, 14)))
        #print("Passed Stage 3 Concat ShuffleUnit test.")


    def test_stage2_add(self):
        groups, in_channels, out_channels = 3, 240, 240
        x = get_input(num_channels=in_channels, height=28, width=28)
        unit = ShuffleUnit(
            in_channels,
            out_channels,
            groups=groups,
            grouped_conv=True,
            combine='add'
            )
        out = unit.forward(x)

        self.assertEqual(0, np.any(out.data.size() != (1, 240, 28, 28)))
        #print("Passed Stage 2 Add ShuffleUnit test.")


    def test_stage2_firstShuffleUnit(self):
        groups = 3
        in_channels = 24
        out_channels = 240
        x = get_input(num_channels=in_channels, height=56, width=56)
        unit = ShuffleUnit(
            in_channels,
            out_channels,
            groups=groups,
            grouped_conv=False,
            combine='concat'
            )

        out = unit.forward(x)
        self.assertEqual(0, np.any(out.data.size() != (1, 240, 28, 28)))


class TestChannelShuffle(unittest.TestCase):
    def test(self):
        
        batchsize = 1
        num_channels = 4
        height = 2
        width = 2
        groups = 2
        
        # prepare inputs
        shape = (batchsize, num_channels, height, width)
        tensor = torch.FloatTensor(
            np.arange(np.product(shape)).astype(np.float32).reshape(shape))
        x = Variable(tensor)

        # run function
        out = channel_shuffle(x, groups).data.numpy()

        # true answer
        answer =  np.array([0,   1,
                            2,   3,
                            8,   9,
                           10,  11,
                            4,   5,
                            6,   7,
                           12,  13,
                           14,  15]).reshape(shape)
        self.assertEqual(0, np.any(out != answer))
        #print("Passed channel shuffle test.")


class TestShuffleNet(unittest.TestCase):
    def test(self):
        # ImageNet image input size
        x = get_input(batchsize=1, num_channels=3, width=224, height=224)
        num_classes = 1000
        groups = 3
        net = ShuffleNet(groups=groups, num_classes=num_classes)
        out = net.forward(x)
        self.assertEqual(0, np.any(out.data.size() != (1, 1000)))


if __name__ == "__main__":
    unittest.main()