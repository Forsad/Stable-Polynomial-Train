from src.activations.poly_activation import HEActivation
from torch import nn
import torch
class MLP(nn.Module):
    def __init__(self, activation=None, activation_params=None, 
                 input_size=None, num_classes=None, dropout=None,
                bn_before_act=None, use_singleton_activation=None,
                hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        self.activation = activation
        self.activation_params = activation_params or {}
        self.dropout = dropout
        self.bn_before_act = bn_before_act
        self.use_singleton_activation = use_singleton_activation
        # Feature processing
        self.singleton_activation = None
        if self.use_singleton_activation:
            self.singleton_activation = HEActivation(self.activation, self.activation_params)
        self.features = nn.Sequential()
        in_dim = input_size
        
        for i, out_dim in enumerate(hidden_dims):
            self.features.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            cur_activaiton = self.singleton_activation if self.use_singleton_activation else HEActivation(self.activation, self.activation_params)
            if self.bn_before_act:
                self.features.add_module(f"bn_{i}", nn.BatchNorm1d(out_dim))
                self.features.add_module(f"act_{i}", cur_activaiton)
            else:
                self.features.add_module(f"act_{i}", cur_activaiton)
                self.features.add_module(f"bn_{i}", nn.BatchNorm1d(out_dim))


            if self.dropout > 0:
                self.features.add_module(f"drop_{i}", nn.Dropout(self.dropout))
            in_dim = out_dim
            
        # Classifier with dimension reduction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            self.activation if self.use_singleton_activation else HEActivation(self.activation, self.activation_params),
            nn.Dropout(self.dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, channels, timesteps)
        x = x.view(x.size(0), -1)  # Flatten sensor channels and timesteps
        x = self.features(x)
        return self.classifier(x)

    def fold_batchnorm(self):
        """Fold BatchNorm layers into Linear layers for HE compatibility"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                path = name.split('.')
                parent = self
                for p in path[:-1]:
                    parent = getattr(parent, p)
                bn_name = f"bn_{path[-1].split('_')[-1]}"  # Match bn to linear index
                
                if hasattr(parent, bn_name):
                    bn = getattr(parent, bn_name)
                    gamma = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                    module.weight.data *= gamma.view(-1, 1)
                    if module.bias is not None:
                        module.bias.data = (module.bias - bn.running_mean) * gamma + bn.bias
                    else:
                        module.bias = nn.Parameter((-bn.running_mean) * gamma + bn.bias)
                    setattr(parent, bn_name, nn.Identity())