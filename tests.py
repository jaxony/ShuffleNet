import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model import channel_shuffle


class TestChannelShuffle(unittest.TestCase):
    def test(self):
        
        batchsize = 1
        num_channels = 4
        height = 2
        width = 2
        groups = 2
        
        # input shape
        shape = (batchsize, num_channels, height, width)

        # prepare inputs
        tensor = torch.FloatTensor(np.arange(np.product(shape)).reshape(shape))
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
        print("Passed channel shuffle test.")


if __name__ == "__main__":
    unittest.main()