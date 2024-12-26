import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, level, input_shape, conditional, num_classes, width="standard"):
        super(Decoder, self).__init__()
        
        # Define the model width variations
        model_width = {
            "narrow": [8, 16, 32],
            "standard": [16, 32, 64],
            "wide": [32, 64, 128]
        }
        
        if width not in {"narrow", "standard", "wide"}:
            raise ValueError("width must be one of {'narrow', 'standard', 'wide'}")
        
        self.widths = model_width[width]
        
        # Conditional input handling
        self.conditional = conditional
        if self.conditional:
            self.embedding = nn.Embedding(num_classes, 50)  # Embed label to vector
            self.fc = nn.Linear(50, input_shape[1] * input_shape[2])  # Fully connected to match shape
            self.fc_out_channels = input_shape[1] * input_shape[2]
        self.in_c = input_shape[0] + 1 if self.conditional else input_shape[0]
        self.level = level
        self.layers = nn.ModuleList()
        # add layers from bottom to top
        layer_config = {
            1 : (self._build_conv_block, [self.widths[0], self.widths[0]]),
            2 : (self._build_conv_block, [self.widths[0], self.widths[0]]),
            3 : (self._build_conv_block, [self.widths[1], self.widths[0]]),
            4 : (self._upsample_block  , [self.widths[1], self.widths[1]]),
            5 : (self._build_conv_block, [self.widths[1], self.widths[1]]),
            6 : (self._build_conv_block, [self.widths[2], self.widths[1]]),
            7 : (self._upsample_block  , [self.widths[2], self.widths[2]]),
            8 : (self._build_conv_block, [self.widths[2], self.widths[2]]),
            9 : (self._build_conv_block, [self.widths[2], self.widths[2]]),

        }
        for i in range(1, level+1):
            mehtod, [in_c, out_c] = layer_config[i]
            self.layers = mehtod(self.in_c if i == level else in_c, out_c) + self.layers
        self.final_conv = nn.Conv2d(self.widths[0], 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        

    def _build_conv_block(self, in_channels, out_channels):
        """Creates a Conv2D -> BatchNorm -> LeakyReLU block."""
        layers = nn.ModuleList()
        layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
        layers.append(nn.ReLU())
        return layers
    
    def _upsample_block(self, in_channels, out_channels):
        """Creates an UpSampling2D -> Conv2D block."""
        layers = nn.ModuleList()
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
        layers.append(nn.ReLU())
        return layers

    def forward(self, xin, yin=None):
        # Conditional embedding
        if self.conditional:
            yin_embedding = self.embedding(yin)  # Get the embedded label
            yin_embedding = self.fc(yin_embedding)  # Pass through a fully connected layer
            yin_embedding = yin_embedding.view(-1, 1, xin.size(2), xin.size(3))  # Reshape to match input
            xin = torch.cat([xin, yin_embedding], dim=1)  # Concatenate conditional input with image input
        x = xin
        # Pass through the decoder layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)
        return self.sigmoid(x)


class SimulatorDiscriminator(nn.Module):
    def __init__(self, level, input_shape, conditional, num_class, dataset_name, width="standard", bn=True):
        super(SimulatorDiscriminator, self).__init__()
        self.dataset_name = dataset_name
        
        # Define the model width variations
        model_width = {
            "narrow": [16, 32, 64, 128],
            "standard": [32, 64, 128, 256],
            "wide": [64, 128, 256, 512]
        }
        
        if width not in {"narrow", "standard", "wide"}:
            raise ValueError("width must be one of {'narrow', 'standard', 'wide'}")
        
        self.widths = model_width[width]
        
        # Conditional input handling
        self.conditional = conditional
        self.in_channels = input_shape[0] + 1 if conditional else input_shape[0]
        if self.conditional:
            self.embedding = nn.Embedding(num_class, 50)  # Embed label to vector
            self.fc = nn.Linear(50, input_shape[1] * input_shape[2])  # Fully connected to match shape
        
        # Batch normalization flag
        self.bn = bn
        
        # Define convolution layers based on the level
        self.conv_layers = nn.ModuleList()
        scale_fc = 32 if self.dataset_name in {"cifar10", "cifar100"} else 16 if self.dataset_name == "tinyimagenet" else 32/3
        if level == 3:  # input_shape = (32, 32, 16)
            self.conv_layers.extend(self._build_level_3())
        elif level <= 6:
            self.conv_layers.extend(self._build_level_6())
        elif level <= 9:
            self.conv_layers.extend(self._build_level_9())
        
        # Fully connected layers for classifier

        self.fc1 = nn.Linear(int(self.widths[3] * input_shape[0] * input_shape[1] // scale_fc), 1)
        self.dropout = nn.Dropout(0.4)

    def _build_level_3(self):
        layers = nn.ModuleList()
        layers.extend(self._build_conv_block(self.in_channels, self.widths[0], stride = 1))
        layers.extend(self._build_conv_block(self.widths[0], self.widths[1], check_bn=True, stride = 2))
        layers.extend(self._build_conv_block(self.widths[1], self.widths[2], check_bn=True, stride = 2))
        layers.extend(self._build_conv_block(self.widths[2], self.widths[3], check_bn=True, stride = 1))
        layers.extend(self._build_conv_block(self.widths[3], self.widths[3], check_bn=True, stride = 1))
        layers.extend(self._build_conv_block(self.widths[3], self.widths[3], check_bn=True, stride = 1))
        layers.append(nn.Conv2d(self.widths[3], self.widths[3], kernel_size=3, stride=2, padding=1))
        return layers
    
    def _build_level_6(self):
        layers = nn.ModuleList()
        layers.extend(self._build_conv_block(self.in_channels, self.widths[1], stride = 1))
        layers.extend(self._build_conv_block(self.widths[1], self.widths[2], check_bn=True, stride = 2))
        layers.extend(self._build_conv_block(self.widths[2], self.widths[3], check_bn=True, stride = 1))
        layers.extend(self._build_conv_block(self.widths[3], self.widths[3], check_bn=True, stride = 1))
        layers.extend(self._build_conv_block(self.widths[3], self.widths[3], check_bn=True, stride = 1))
        layers.append(nn.Conv2d(self.widths[3], self.widths[3], kernel_size=3, stride=2, padding=1))
        return layers
    
    def _build_level_9(self):
        layers = nn.ModuleList()
        layers.extend(self._build_conv_block(self.in_channels, self.widths[2], check_bn=False, stride = 1))
        layers.extend(self._build_conv_block(self.widths[2], self.widths[3], check_bn=True, stride = 1))
        layers.extend(self._build_conv_block(self.widths[3], self.widths[3], check_bn=True, stride = 1))
        layers.extend(self._build_conv_block(self.widths[3], self.widths[3], check_bn=True, stride = 1))
        layers.append(nn.Conv2d(self.widths[3], self.widths[3], kernel_size=3, stride=2, padding=1))
        return layers

    def _build_conv_block(self, in_channels, out_channels, check_bn=False, stride=1):
        """Creates a Conv2D -> BatchNorm -> LeakyReLU block."""
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1))
        if check_bn and self.bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
        layers.append(nn.LeakyReLU(0.2))
        return layers

    def forward(self, xin, yin=None):
        # Conditional input
        if self.conditional:
            yin_embedding = self.embedding(yin)  # Get the embedded label
            yin_embedding = self.fc(yin_embedding)  # Fully connected to match shape
            yin_embedding = yin_embedding.view(-1, 1, xin.size(2), xin.size(3))  # Reshape to match input
            xin = torch.cat([xin, yin_embedding], dim=1)  # Concatenate conditional input with image input
        
        # Pass through the convolution layers
        x = xin

        for layer in self.conv_layers:
            x = layer(x)
        # Flatten and pass through the classifier
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)  # Dropout
        x = self.fc1(x)  # Final classifier layer
        return x

class DecoderDiscriminator(nn.Module):
    def __init__(self, input_shape, conditional, num_class):
        super(DecoderDiscriminator, self).__init__()
        self.input_shape = input_shape
        # Conditional input handling
        self.conditional = conditional
        if self.conditional:
            self.embedding = nn.Embedding(num_class, 50)  # Embed label to vector
            self.fc = nn.Linear(50, input_shape[1] * input_shape[2])  # Fully connected to match shape
        
        # Define the convolution layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0] if not conditional else input_shape[0] + 1, 
                               out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Batch normalization layers
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers for classifier
        self.fc1 = nn.Linear(256 * input_shape[1] // 8 * input_shape[2] // 8, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, xin, yin=None):
        if self.conditional:
            # Embed and process the label (conditional input)
            yin_embedding = self.embedding(yin)  # Get the embedded label. Shape BatchSize x (32*32)
            yin_embedding = self.fc(yin_embedding)  # Fully connected to match shape
            yin_embedding = yin_embedding.view(-1, 1, xin.size(2), xin.size(3))  # Reshape to match input size
            xin = torch.cat([xin, yin_embedding], dim=1)  # Concatenate the conditional label with the input image
        # Convolutional layers with LeakyReLU activations
        x = F.leaky_relu(self.conv1(xin), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        
        # Flatten the output and apply dropout
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)  # Dropout layer
        x = self.fc1(x)  # Final classifier layer
        
        return x