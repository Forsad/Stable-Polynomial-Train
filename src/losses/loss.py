import torch
from torch import nn
from src.activations.poly_activation import PolyActivation

class CustomPolyLoss(nn.Module):
    def __init__(self, base_criterion, lambda_penalty, enable_boundary_loss,
                 boundary_loss_params):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda_penalty = lambda_penalty
        self.enable_boundary_loss = enable_boundary_loss
        if enable_boundary_loss:
            self.b_type = boundary_loss_params['type']
            self.acc_norm = boundary_loss_params['acc_norm']
            if self.b_type in ['exp', 'l2']:
                if self.acc_norm != 'sum':
                    raise ValueError(f"Unknown norm type: {self.acc_norm}")
                if 'penalty_B' not in boundary_loss_params:
                    raise ValueError("penalty_B is required for exp boundary loss")
            elif self.b_type == 'maxnorm':
                self.acc_norm = boundary_loss_params['acc_norm']
                if self.acc_norm not in [1, 2, 'max']:
                    raise ValueError(f"Unknown norm type: {self.acc_norm}")
            else:
                raise ValueError(f"Unknown boundary loss type: {self.b_type}")
    
    def forward(self, outputs, labels, model):
        # Calculate base classification loss
        base_loss = self.base_criterion(outputs, labels)
        
        # Calculate boundary penalty from all PolyActivation layers
        boundary_loss = 0
        boundary_loss_list = []
        for module in model.modules():
            if isinstance(module, PolyActivation) and hasattr(module, 'current_boundary_loss'):
                boundary_loss_list.append(module.current_boundary_loss)
        
        
        total_loss = base_loss
        # #total_loss = base_loss + self.lambda_penalty * boundary_loss + self.lambda_penalty * kan_boun_loss
        if self.enable_boundary_loss:
            boundary_loss_list = torch.stack(boundary_loss_list)
            if self.acc_norm == 'sum':
                boundary_loss = torch.sum(boundary_loss_list)
            elif self.acc_norm == 'max':
                boundary_loss = torch.max(boundary_loss_list)
            elif self.acc_norm in [1, 2]:
                boundary_loss = torch.norm(boundary_loss_list, p=self.acc_norm)
                
            total_loss += self.lambda_penalty * boundary_loss
        return total_loss, base_loss, boundary_loss, 0.0