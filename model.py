import torch
import torch.nn as nn
from torch.autograd import Variable


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    """3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride,
        padding=padding, 
        bias=bias)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        groups=groups,
        stride=1)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups,
                 grouped_conv=True, depthwise_stride=1):
        
        super(ShuffleUnit, super).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.depthwise_stride = depthwise_stride

        self.groups = groups if grouped_conv else 1
        self.bottleneck_channels = self.out_channels // 4

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            batch_norm=True,
            relu=True
            )


    def _make_grouped_conv1x1(self, in_channels, out_channels,
        batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=self.groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['batch_norm'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv




    def _combine(operation):
        if operation == 'add':
            pass
        elif operation == 'concat':
            pass
        else:
            raise ValueError("operation \"{}\" not supported.".format(operation))

    def forward(self, x):
        pass


class ShuffleNet(nn.Module):
    """ 
    ShuffleNet implementation.
    """

    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped 
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.

        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels =  in_channels
        self.num_classes = num_classes

        if groups == 1:
            self.stage_out_channels = [0, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [0, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [0, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [0, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [0, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for 
                   1x1 Grouped Convolutions""".format(num_groups))
        
        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels, 
                             self.stage_out_channels[0], 
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)

        # Stage 3
        self.stage3 = self._make_stage(3)

        # Stage 4
        self.stage4 = self._make_stage(4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)


    def _make_stage(self, stage):
        layers = []
        
        # first layer is special
        # - non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in stage 2. Group convolutions used thereafter.
        # - depthwise convolution has a stride of 2 for all stages.
        grouped_conv = stage > 2
        first_layer = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            depthwise_stride=2,
            grouped_conv=grouped_conv,
            )
        layers.append(first_layer)

        # add more ShuffleUnits depending on pre-defined number of repeats
        for _ in range(self.stage_repeats[stage-2]):
            layer = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                grouped_conv=True,
                )
            
            layers.append(layer)

        return layers


    def forward(self, x):
        pass


if __name__ == "__main__":
    """ Testing
    """
    model = ShuffleNet()
    print(model)
