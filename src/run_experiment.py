import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainers.lightning_model import LightningModel, LREpochEndCallback, AnyNanTermination
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
import torch
import wandb

def run_expriment(*, project_params:dict, dataset_params:dict, model_params:dict, training_params:dict):
    run_id = project_params['run_id']
    project_name = project_params['project_name']
    custom_tag = project_params['custom_tag']

    # Extract dataset parameters
    data_workers = dataset_params['data_workers']
    dataset = dataset_params['dataset']
    batch_size = dataset_params['batch_size']

    config = {
        'training_params': training_params,
        'data_workers': data_workers,
        'batch_size': batch_size,
        'model_params': model_params,
        'custom_tag': custom_tag
    }
    api_key_present = os.environ.get("WANDB_API_KEY")
    if api_key_present is not None:
        wandb.finish() 
        wandb.init(project=project_name, name=run_id, config=config)
    else:
        wandb.init(project=project_name, name=run_id, config=config, mode='disabled')

    train_set = dataset['train']
    val_set = dataset['val']
    test_set = dataset['test']


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=data_workers)
    val_loader = DataLoader(val_set, batch_size=2*batch_size, shuffle=False, pin_memory=True, num_workers=data_workers)
    test_loader = DataLoader(test_set, batch_size=2*batch_size, shuffle=False, pin_memory=True, num_workers=data_workers)
    
    model = LightningModel(training_params=training_params, model_params=model_params)
    model.cuda()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_epoch',    # Must match the logged metric name
        mode='max',           # Maximize validation accuracy
        save_top_k=1,
        filename='best-acc-{epoch}-{val_acc:.2f}'
    )
    # Trainer
    max_epoch = training_params['max_epoch']
    gradient_clip_val = training_params['gradient_clip_val']

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator='gpu',
        gradient_clip_val=gradient_clip_val,
        callbacks=[LearningRateMonitor(logging_interval='epoch'),
                LREpochEndCallback(), AnyNanTermination(), checkpoint_callback],
    )
    
    torch.autograd.set_detect_anomaly(True)

    trainer.fit(model, train_loader, val_loader)

    print("Training completed!")
    ret_val = trainer.validate(ckpt_path='best', dataloaders=val_loader)
    ret_test = trainer.validate(ckpt_path='best', dataloaders=test_loader)
    wandb.run.summary["val_acc"] = ret_val[0]['val_acc_epoch']
    wandb.run.summary["val_loss"] = ret_val[0]['val_loss_epoch']
    wandb.run.summary["test_acc"] = ret_test[0]['val_acc_epoch']
    wandb.run.summary["test_loss"] = ret_test[0]['val_loss_epoch']
    wandb.finish()
    trainer.save_checkpoint(f"saved_models/model_{run_id}.ckpt")