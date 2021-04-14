import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch
from torch.nn import Parameter
from .RSG import *

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class ResNeXtBottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
    super(ResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality

    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_reduce = nn.BatchNorm2d(D*C)

    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.bn = nn.BatchNorm2d(D*C)

    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_expand = nn.BatchNorm2d(planes*4)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):

  def __init__(self, block, depth, cardinality, base_width, num_classes, use_norm=False, head_tail_ratio = 0.3, transfer_strength = 1.0, phase_train=True, epoch_thresh=0):
    super(CifarResNeXt, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 9 == 0
    layer_blocks = (depth - 2) // 9

    self.cardinality = cardinality
    self.base_width = base_width
    self.num_classes = num_classes
    self.phase_train = phase_train
    self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    self.bn_1 = nn.BatchNorm2d(64)

    self.inplanes = 64
    self.stage_1 = self._make_layer(block, 64 , layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)
    if use_norm:
            self.classifier = NormedLinear(256*block.expansion, num_classes)
    else:
            self.classifier = nn.Linear(256*block.expansion, num_classes)
    if self.phase_train:
           self.head_lists = [x for x in range(int(num_classes * head_tail_ratio))]
           self.RSG = RSG(n_center = 15, feature_maps_shape = [512, 16, 16], num_classes=num_classes, contrastive_module_dim = 256, head_class_lists = self.head_lists, transfer_strength = transfer_strength, epoch_thresh = epoch_thresh)
 
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

    return nn.Sequential(*layers)

  def forward(self, x, epoch=0, batch_target=None,  phase_train=True):
    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1(x), inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)

    if phase_train:
          x, cesc_total, loss_mv_total, combine_target = self.RSG.forward(x, self.head_lists, batch_target, epoch)

    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    out = self.classifier(x)
    
    if phase_train:
          return out, cesc_total, loss_mv_total, combine_target
    else:
          return out
          


def resnext29_8_64(num_classes=10, use_norm=False, head_tail_ratio = 0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
  """Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 8, 64, num_classes, use_norm=use_norm, head_tail_ratio = head_tail_ratio, transfer_strength = transfer_strength, phase_train=phase_train, epoch_thresh=epoch_thresh)
  return model

