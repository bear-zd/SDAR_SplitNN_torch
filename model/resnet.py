import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample+1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
 
        if downsample:
            self.downsampleconv  = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            self.downsamplebn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsampleconv(identity)
            identity = self.downsamplebn(identity)
        out += identity
        out = self.relu(out)
        return out

class FModel(nn.Module):
    def __init__(self, level, input_shape, width="standard"):
        super(FModel, self).__init__()

        if width not in ["narrow", "standard", "wide"]:
            raise ValueError(f'Width {width} is not supported.')
        model_width = {
            "narrow": [8, 16, 32],
            "standard": [16, 32, 64],
            "wide": [32, 64, 128]
        }
        self.widths = model_width[width]
    
        layer_config = {
            1 : [self.widths[0], self.widths[0]],
            2 : [self.widths[0], self.widths[0]],
            3 : [self.widths[0], self.widths[0]],
            4 : [self.widths[0], self.widths[1]],
            5 : [self.widths[1], self.widths[1]],
            6 : [self.widths[1], self.widths[1]],
            7 : [self.widths[1], self.widths[2]],
            8 : [self.widths[2], self.widths[2]],
            9 : [self.widths[2], self.widths[2]],
        }
        
        if level < 3 or level > 9:
            raise NotImplementedError(f'Level {level} is not supported.')
        # init pre resblock ops
        self.conv1 = nn.Conv2d(input_shape[0], self.widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.widths[0], momentum=0.9, eps=1e-5, affine=False)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(level, 0, -1):
            in_c, out_c = layer_config[i]
            if i == 0:
                in_c = input_shape[0]
            self.layers.insert(0, BasicBlock(in_c, out_c, True if i in [4, 7] else False))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        return x

class GModel(nn.Module):
    def __init__(self, level, input_shape, num_classes=10, dropout=0.0, width="standard"):
        super(GModel, self).__init__()

        if width not in ["narrow", "standard", "wide"]:
            raise ValueError(f'Width {width} is not supported.')
        model_width = {
            "narrow": [8, 16, 32],
            "standard": [16, 32, 64],
            "wide": [32, 64, 128]
        }
        if level < 3 or level > 9:
            raise NotImplementedError(f'Level {level} is not supported.')
        self.widths = model_width[width]
        layer_config = {
            3 : [self.widths[0], self.widths[1]],
            4 : [self.widths[1], self.widths[1]],
            5 : [self.widths[1], self.widths[1]],
            6 : [self.widths[1], self.widths[2]],
            7 : [self.widths[2], self.widths[2]],
            8 : [self.widths[2], self.widths[2]],
            9 : [self.widths[2], self.widths[2]],
        }

        self.layers = nn.ModuleList()
        for i in range(8, level-1, -1):
            in_c, out_c  = layer_config[i]
            self.layers.insert(0, BasicBlock(in_c if i != level else input_shape[0], out_c, True if i in [3, 6] else False))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.widths[2], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x).squeeze(2).squeeze(2)
        x = self.fc(x)
        x = self.dropout(x)
        return x