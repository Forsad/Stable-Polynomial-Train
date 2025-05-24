import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from src.losses.loss import CustomPolyLoss
from src.activations.poly_activation import EpochAwareModule
from src.models.resnet import ResNet18
import wandb
import torch
from pytorch_lightning.callbacks import Callback
from src.models.mlp import MLP
    
    
class LightningModel(pl.LightningModule):
    def __init__(self, model_params:dict, training_params:dict):
        super().__init__()
        input_size = model_params['input_size']
        model = model_params['model']
        activation = model_params['activation']
        activation_params = model_params['actvation_params']
        num_classes = model_params['num_classes']
        dropout = model_params['dropout']
        bn_before_act = model_params['bn_before_act']
        use_singleton_activation = model_params['use_singleton_activation']
        if model == 'resnet18':
            self.model = ResNet18(activation=activation, activation_params=activation_params, num_classes=num_classes,
                              dropout=dropout, input_size=input_size, bn_before_act=bn_before_act,
                              use_singleton_activation=use_singleton_activation)
        elif model == 'mlp':
            hidden_dims = model_params['hidden_dims']
            self.model = MLP(activation=activation, activation_params=activation_params, num_classes=num_classes,
                                      dropout=dropout, input_size=input_size, bn_before_act=bn_before_act,
                                      use_singleton_activation=use_singleton_activation, hidden_dims=hidden_dims)
        else:
            raise ValueError(f"Unsupported model: {model}")


        enable_boundary_loss = training_params['enable_boundary_loss']
        lambda_penalty = training_params['lambda_penalty']
        disable_batchnorm_grad_clip_exclusion = training_params['disable_batchnorm_grad_clip_exclusion']

        self.optimizer_params = training_params['optimizer_params']
        self.scheduler_params = training_params['scheduler_params']

        self.base_criterion = nn.CrossEntropyLoss()
        self.boundary_loss_params = activation_params['boundary_loss_params']
        self.criterion = CustomPolyLoss(self.base_criterion,lambda_penalty, enable_boundary_loss, self.boundary_loss_params)
        self.val_loss = 0
        self.css_loss = 0
        self.b_loss = 0
        self.kan_loss = 0
        self.total = 0
        self.correct = 0
        self.bn_params = set()
        self.disable_batchnorm_grad_clip_exclusion = disable_batchnorm_grad_clip_exclusion
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.bn_params.update(module.parameters())
        

    def forward(self, x):
        return self.model(x)
    #Turning off gradient clipping for now for experiments; will be used in the future
    def configure_gradient_clipping(self,  optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Only clip non-BatchNorm parameters
        if gradient_clip_val is not None:
            if self.disable_batchnorm_grad_clip_exclusion:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
            else:
                non_bn_params = [p for p in self.parameters() if p not in self.bn_params]
                torch.nn.utils.clip_grad_norm_(non_bn_params, gradient_clip_val)


    def on_train_epoch_start(self):
        # Update epoch for all submodules
        for module in self.modules():
            if isinstance(module, EpochAwareModule):
                module.epoch = self.current_epoch

    def on_validation_epoch_start(self):
        self.val_loss = 0
        self.css_loss = 0
        self.b_loss = 0
        self.kan_loss = 0
        self.total = 0
        self.correct = 0

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
    
        outputs = self(inputs)
        loss, cross_entropy_loss, boundary_loss, _ = self.criterion(outputs, labels, self.model)
        boun_loss = boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_boundary_loss', boundary_loss, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train_loss": loss.item() * len(labels),
                   "train_cross_entropy_loss": cross_entropy_loss.item(),
                   "train_boundary_loss": boun_loss
                   })
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss, css_loss, b_loss, _ = self.criterion(outputs, labels, self.model)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        self.val_loss +=  loss.item() * len(labels)
        self.css_loss += css_loss.item() * len(labels)
        if isinstance(b_loss, torch.Tensor):
            self.b_loss += b_loss.item() * len(labels)
        self.total += len(labels)
        self.correct += correct

        return {'val_loss': loss * len(labels), 'correct': correct, 'total': len(labels)}
    


    def on_validation_epoch_end(self):

        self.log('val_loss_epoch', self.val_loss / self.total, on_epoch=True, prog_bar=True)
        self.log('val_acc_epoch', self.correct / self.total, on_epoch=True, prog_bar=True)
        self.log('val_cross_entropy_loss_epoch', self.css_loss / self.total, on_epoch=True, prog_bar=True)
        self.log('val_boundary_loss_epoch', self.b_loss / self.total, on_epoch=True, prog_bar=True)

        wandb.log({"val_loss": self.val_loss / self.total,
                   "val_acc": self.correct / self.total,
                   "val_cross_entropy_loss": self.css_loss / self.total,
                   "val_boundary_loss": self.b_loss / self.total
                   })
            

    def configure_optimizers(self):
        poly_params = []
        other_params = []
        for name, param in self.named_parameters():
            if 'coefficients' in name and param.requires_grad:
                poly_params.append(param)
            else:
                other_params.append(param)
        optimizer = None
        scheduler = None
        learning_rate = self.optimizer_params['lr']
        if self.optimizer_params['type'] == 'adamw':
            optimizer = optim.AdamW([
                {'params': poly_params, 'lr': 0.1 * learning_rate},
                {'params': other_params, 'lr': learning_rate}
            ], **self.optimizer_params['params'])
        elif self.optimizer_params['type'] == 'sgd':
            optimizer = optim.SGD([
                {'params': poly_params, 'lr': 0.1 * learning_rate},
                {'params': other_params, 'lr': learning_rate}
            ], **self.optimizer_params['params'])
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_params['type']}")
        
        scheduler = None
        if self.scheduler_params['type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params['params'])
        elif self.scheduler_params['type'] == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_params['params'])
        elif self.scheduler_params['type'] == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_params['params'])
        elif self.scheduler_params['type'] == 'cosine_annealing_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params['params'])
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_params['type']}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss_epoch'
        }




class LREpochEndCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Get current learning rate
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({"learning_rate": current_lr})
        pl_module.log('learning_rate', current_lr, prog_bar=False)

class AnyNanTermination(Callback):
    def __init__(self, monitors=('train_loss', 'val_loss')):
        self.monitors = monitors

    def on_train_epoch_end(self, trainer, pl_module):
        self._check_metrics(trainer, phase="training")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._check_metrics(trainer, phase="validation")

    def _check_metrics(self, trainer, phase):
        for metric in self.monitors:
            value = trainer.callback_metrics.get(metric)
            if value is not None and not torch.isfinite(value):
                print(f"\nStopping due to {metric}={value} in {phase} phase")
                trainer.should_stop = True
                return 


