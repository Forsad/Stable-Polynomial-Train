from src.activations.poly_activation import PolyFit, HEActivation
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int, activation:str,
                 activation_params:dict, dropout:float, bn_before_act:bool,
                 singleton_activation: PolyFit | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.activation_type = activation

        if singleton_activation is not None:
            self.used_singleton_activation = True
            self.act1 = singleton_activation
        else:
            self.used_singleton_activation = False
            self.act1 = HEActivation(activation, activation_params)
            
        self.bn_before_act = bn_before_act

        if self.bn_before_act:
            self.bn_act1 = nn.Sequential(self.bn1, self.act1)
        else:
            self.bn_act1 = nn.Sequential(self.act1, self.bn1)
        
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()



        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if singleton_activation is not None:    
            self.act2 = singleton_activation
        else:
            self.act2 = HEActivation(activation, activation_params)

        
        if self.bn_before_act:
            self.bn_act2 = nn.Sequential(self.bn2, self.act2)
        else:
            self.bn_act2 = nn.Sequential(self.act2, self.bn2)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn_act1(out)
        out = self.dropout1(out)
        out = self.conv2(out)

        out += self.shortcut(x)
        out = self.bn_act2(out)
        out = self.dropout2(out)
        return out



class ResNet18(nn.Module):
    """
    ResNet18 model with configurable activation function and parameters
    activation: str, activation function name
    activation_params: dict, activation function parameters
    num_classes: int, number of classes
    dropout: float, dropout rate
    input_size: tuple[int, int], input size
    bn_before_act: bool, whether to apply batch normalization before activation
    use_singleton_activation: bool, whether to use singleton (i.e. a single activation function for all layers)
    """
    def __init__(self, *, activation:str, activation_params:dict, num_classes:int, dropout:float, input_size:tuple[int, int, int],
                 bn_before_act:bool, use_singleton_activation:bool):
        super().__init__()
        self.activation = activation
        self.activation_params = activation_params or {}
        self.dropout_value = dropout
        self.bn_before_act = bn_before_act
        #Right now bn_before_act true with poly activation
        #if activation == 'poly' and not bn_before_act:
            #raise ValueError("Poly activation must have bn_before_act true")

        if use_singleton_activation:
            self.primary_act = HEActivation(activation, activation_params)
        else:
            self.primary_act = None
        
        self.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        if use_singleton_activation:
            self.act1 = self.primary_act
        else:
            self.act1 = HEActivation(activation, activation_params)

        if self.bn_before_act:
            self.bn_act1 = nn.Sequential(self.bn1, self.act1)
        else:
            self.bn_act1 = nn.Sequential(self.act1, self.bn1)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        
        # Dropout layer for regularization
        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)
        
    def initialize_dynamic_properties(self, input_size=(32, 32)):
        # Call this after moving to CUDA
        self.eval()  # Ensure we're in eval mode for this operation
        with torch.no_grad():
            rand_input = torch.randn(1, 3, input_size[0], input_size[1], device=self.device)
            self.feature_map_size = self.forward_conv_layers(rand_input).shape[2:]
            self.avg_pool_factor = 1.0 / (self.feature_map_size[0] * self.feature_map_size[1])
        self.train()


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(
                in_channels, 
                out_channels, 
                stride, 
                self.activation, 
                self.activation_params,
                self.dropout_value,
                self.bn_before_act,
                self.primary_act
            ))
            in_channels = out_channels
        return nn.Sequential(*layers)
    def forward_conv_layers(self, x):
        out = self.bn_act1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
    def forward(self, x):
        out = self.forward_conv_layers(x)
        #print("Done with conv layers")
        feature_map_size = out.shape[2:]
        out = out.sum(dim=(2, 3), keepdim=True) / (feature_map_size[0] * feature_map_size[1])
        #print("Done with avg pool")
        out = out.view(out.size(0), -1)
        #print("Done with view")
        out = self.final_dropout(out)
        #print("Done with dropout")
        out = self.fc(out)
        #print("Done with fc")
        return out



