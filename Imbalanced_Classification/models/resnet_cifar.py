import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from .RSG import *

__all__ = ['resnet32',  'resnet56', 'resnet110']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, head_tail_ratio = 0.3, transfer_strength = 1.0, phase_train=True, epoch_thresh=0):
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        self.phase_train = phase_train
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

        if self.phase_train:
           self.head_lists = [x for x in range(int(num_classes * head_tail_ratio))]
           self.RSG = RSG(n_center = 15, feature_maps_shape = [32, 16, 16], num_classes=num_classes, contrastive_module_dim = 256, head_class_lists = self.head_lists, transfer_strength = transfer_strength, epoch_thresh = epoch_thresh)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, epoch=0, batch_target=None, phase_train=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)

        if phase_train:
          out, cesc_total, loss_mv_total, combine_target = self.RSG.forward(out, self.head_lists, batch_target, epoch)

        out = self.layer3(out)     
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if phase_train:
          return out, cesc_total, loss_mv_total, combine_target
        else:
          return out


def resnet32(num_classes=10, use_norm=False, head_tail_ratio = 0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, head_tail_ratio = head_tail_ratio, transfer_strength = transfer_strength, phase_train=phase_train, epoch_thresh=epoch_thresh)



def resnet56(num_classes=10, use_norm=False, head_tail_ratio = 0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
    return ResNet_s(BasicBlock, [9, 9, 9], num_classes=num_classes, use_norm=use_norm, head_tail_ratio = head_tail_ratio, transfer_strength = transfer_strength, phase_train=phase_train, epoch_thresh=epoch_thresh)
    


def resnet110(num_classes=10, use_norm=False, head_tail_ratio = 0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
    return ResNet_s(BasicBlock, [18, 18, 18], num_classes=num_classes, use_norm=use_norm, head_tail_ratio = head_tail_ratio, transfer_strength = transfer_strength, phase_train=phase_train, epoch_thresh=epoch_thresh)



if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
