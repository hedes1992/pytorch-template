import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import pdb

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
