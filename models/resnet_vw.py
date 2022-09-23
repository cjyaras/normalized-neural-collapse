import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
   expansion = 1

   def __init__(self, in_planes, planes, stride=1):
      super(BasicBlock, self).__init__()
      self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(planes)

      self.shortcut = nn.Sequential()
      if stride != 1 or in_planes != self.expansion * planes:
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

class Bottleneck(nn.Module):
   expansion = 4

   def __init__(self, in_planes, planes, stride=1):
      super(Bottleneck, self).__init__()
      self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
      self.bn1 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(planes)
      self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
      self.bn3 = nn.BatchNorm2d(self.expansion * planes)

      self.shortcut = nn.Sequential()
      if stride != 1 or in_planes != self.expansion * planes:
         self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(self.expansion * planes)
         )

   def forward(self, x):
      out = F.relu(self.bn1(self.conv1(x)))
      out = F.relu(self.bn2(self.conv2(out)))
      out = self.bn3(self.conv3(out))
      out += self.shortcut(x)
      out = F.relu(out)
      return out

class ResNetVariableWidth(nn.Module):
   def __init__(self, block, num_blocks, k, num_classes, normalize, fc_bias=False, tau=1):
      super(ResNetVariableWidth, self).__init__()
      self.in_planes = 1 * k

      self.conv1 = nn.Conv2d(3, 1 * k, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(1 * k)
      self.layer1 = self._make_layer(block, 1 * k, num_blocks[0], stride=1)
      self.layer2 = self._make_layer(block, 2 * k, num_blocks[1], stride=2)
      self.layer3 = self._make_layer(block, 4 * k, num_blocks[2], stride=2)
      self.layer4 = self._make_layer(block, 8 * k, num_blocks[3], stride=2)
      self.fc = nn.Linear(8 * k * block.expansion, num_classes, bias = fc_bias)
      self.normalize = normalize
      self.tau = tau

      if self.normalize:
         self.normalize_fc()

   def _make_layer(self, block, planes, num_blocks, stride):
      strides = [stride] + [1] * (num_blocks - 1)
      layers = []
      for stride in strides:
         layers.append(block(self.in_planes, planes, stride))
         self.in_planes = planes * block.expansion
      return nn.Sequential(*layers)

   def forward(self, x):
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = F.avg_pool2d(out, 4)
      out = out.view(out.size(0), -1)
      if self.normalize:
         features = self.tau * F.normalize(out)
      else:
         features = out
      out = self.fc(features)

      return out, features

   @torch.no_grad()
   def normalize_fc(self):
      self.fc.weight.data.copy_(F.normalize(self.fc.weight.data))

def ResNet18VariableWidth(width, **kwargs):
   return ResNetVariableWidth(BasicBlock, [2, 2, 2, 2], k=width, **kwargs)