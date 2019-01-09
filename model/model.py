import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from torchvision.models.resnet import BasicBlock, Bottleneck
import pdb

# copy from resnet.py from torchvision.models
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CloudGermanModel1(BaseModel):
    """
    classification model 1 for cloud german
    """
    def __init__(self, input_channel=18, num_classes=17):
        super(CloudGermanModel1, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2*2*64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 2*2*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CloudGermanModel2(BaseModel):
    """
    classification model 2 for cloud german
    """
    def __init__(self, input_channel=18, num_classes=17):
        super(CloudGermanModel2, self).__init__()
        self.conv1  = nn.Conv2d(input_channel, 32, kernel_size=3)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3  = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.pool4  = nn.MaxPool2d(kernel_size=2)
        self.fc1    = nn.Linear(3*3*64, 128)
        self.fc2    = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x   = F.relu(self.conv1(x))
        x   = F.relu(self.conv2(x))
        x   = F.relu(self.conv3(x))
        x   = self.pool4(x)
        x   = x.view(-1, 3*3*64)
        x   = F.relu(self.fc1(x))
        x   = F.dropout(x, training=self.training)
        x   = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 模仿resnet.py 改造为可以接收 (32,32,18)输入的
class cgResNet(BaseModel):
    def __init__(self, block, layers, input_channel=18, num_classes=17, zero_init_residual=False):
        # 模仿 ResNet
        super(cgResNet, self).__init__()
        self.inplanes   = 64
        self.conv1      = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(64)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
#        self.fc1    = nn.Linear(512 * block.expansion, 64)
#        self.fc2    = nn.Linear(64, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
#        x = self.fc1(x)
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
#        return x

class cgResnet18(cgResNet):
    def __init__(self, input_channel=18, num_classes=17, zero_init_residual=False):
        block   = BasicBlock
        layers  = [2,2,2,2]
        super(cgResnet18, self).__init__(block=block, layers=layers, input_channel=input_channel, \
            num_classes=num_classes, zero_init_residual=zero_init_residual)

class cgResnet34(cgResNet):
    def __init__(self, input_channel=18, num_classes=17, zero_init_residual=False):
        block   = BasicBlock
        layers  = [3,4,6,3]
        super(cgResnet34, self).__init__(block=block, layers=layers, input_channel=input_channel, \
            num_classes=num_classes, zero_init_residual=zero_init_residual)

class cgResnet34_3stage(cgResNet):
    def __init__(self, input_channel=18, num_classes=17, zero_init_residual=False):
        block   = BasicBlock
        layers  = [3,4,3,0]
        super(cgResnet34_3stage, self).__init__(block=block, layers=layers, input_channel=input_channel, \
            num_classes=num_classes, zero_init_residual=zero_init_residual)

class cgResnet34_1stage(cgResNet):
    def __init__(self, input_channel=18, num_classes=17, zero_init_residual=False):
        block   = BasicBlock
        layers  = [3,0,0,0]
        super(cgResnet34_1stage, self).__init__(block=block, layers=layers, input_channel=input_channel, \
            num_classes=num_classes, zero_init_residual=zero_init_residual)

class cgResnet34_0stage(cgResNet):
    def __init__(self, input_channel=18, num_classes=17, zero_init_residual=False):
        block   = BasicBlock
        layers  = [0,0,0,0]
        super(cgResnet34_0stage, self).__init__(block=block, layers=layers, input_channel=input_channel, \
            num_classes=num_classes, zero_init_residual=zero_init_residual)

class cgResnet50(cgResNet):
    def __init__(self, input_channel=18, num_classes=17, zero_init_residual=False):
        block   = Bottleneck
        layers  = [3,4,6,3]
        super(cgResnet50, self).__init__(block=block, layers=layers, input_channel=input_channel, \
            num_classes=num_classes, zero_init_residual=zero_init_residual)
