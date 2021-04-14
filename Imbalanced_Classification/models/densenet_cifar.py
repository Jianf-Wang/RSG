import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .RSG import *

class Bottleneck(nn.Module):
  def __init__(self, nChannels, growthRate):
    super(Bottleneck, self).__init__()
    interChannels = 4*growthRate
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm2d(interChannels)
    self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    out = torch.cat((x, out), 1)
    return out

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class SingleLayer(nn.Module):
  def __init__(self, nChannels, growthRate):
    super(SingleLayer, self).__init__()
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = torch.cat((x, out), 1)
    return out

class Transition(nn.Module):
  def __init__(self, nChannels, nOutChannels):
    super(Transition, self).__init__()
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = F.avg_pool2d(out, 2)
    return out

class DenseNet(nn.Module):
  def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, use_norm=False, head_tail_ratio = 0.3, transfer_strength = 1.0, phase_train=True, epoch_thresh=0):
    super(DenseNet, self).__init__()

    if bottleneck:  nDenseBlocks = int( (depth-4) / 6 )
    else         :  nDenseBlocks = int( (depth-4) / 3 )

    nChannels = 2*growthRate
    self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)

    self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans1 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans2 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate

    self.bn1 = nn.BatchNorm2d(nChannels)
    self.phase_train = phase_train
    if use_norm:
            self.fc = NormedLinear(nChannels, nClasses)
    else:
            self.fc = nn.Linear(nChannels, nClasses)

    if self.phase_train:
           self.head_lists = [x for x in range(int(nClasses * head_tail_ratio))]
           self.RSG = RSG(n_center = 15, feature_maps_shape = [312, 8, 8], num_classes=nClasses, contrastive_module_dim = 256, head_class_lists = self.head_lists, transfer_strength = transfer_strength, epoch_thresh = epoch_thresh)


    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
      if bottleneck:
        layers.append(Bottleneck(nChannels, growthRate))
      else:
        layers.append(SingleLayer(nChannels, growthRate))
      nChannels += growthRate
    return nn.Sequential(*layers)

  def forward(self, x, epoch=0, batch_target=None, phase_train=True):
    out = self.conv1(x)
    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))

    if phase_train:
          out, cesc_total, loss_mv_total, combine_target = self.RSG.forward(out, self.head_lists, batch_target, epoch)

    out = self.dense3(out)
    out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
    out = self.fc(out)
    
    if phase_train:
          return out, cesc_total, loss_mv_total, combine_target
    else:
          return out

def densenet40_12(num_classes=10, use_norm=False, head_tail_ratio = 0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
  model = DenseNet(12, 40, 1.0, num_classes, False, use_norm=use_norm, head_tail_ratio = head_tail_ratio, transfer_strength = transfer_strength, phase_train=phase_train, epoch_thresh=epoch_thresh)
  return model
