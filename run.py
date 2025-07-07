import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint 
from pytorch_lightning.strategies import DDPStrategy 
from dataset import VAEDataset


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()                      # choose config file (vq_vae.yaml)
with open(args.filename, 'r') as file:          # problem: can't open file
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility - obtain seed, model, exp, data, etc info from config file, then train
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])        # exp parameter specific to this "model"

pin_memory = config['trainer_params'].get('accelerator', 'cpu') != 'cpu' 
data = VAEDataset( **config["data_params"], pin_memory=pin_memory ) 

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 log_every_n_steps = 1,                                   # 
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)