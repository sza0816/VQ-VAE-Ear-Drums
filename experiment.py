import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.train_losses = []
        self.val_losses = []

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(
        self, 
        batch, 
        batch_idx
        # optimizer_idx = 0
        ):        ###
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                            #   optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)               # print loss
        # print(f"\n[Epoch {self.current_epoch}][Train] Loss: {train_loss['loss'].item():.4f}") 

        self.training_step_outputs.append(train_loss['loss']) 
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(
        self, 
        batch, 
        batch_idx, 
        # optimizer_idx = 0
        ):      ###
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            # optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        # self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)       # print loss values
        # print(f"\n[Epoch {self.current_epoch}][Val] Loss: {val_loss['loss'].item():.4f}") 

        self.validation_step_outputs.append(val_loss['loss']) 
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True) 

        return val_loss['loss'] 

    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,                                                              ### save image from each epoch
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,                              ### try save image samples to folder, currently has warning
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Exception as e:
            print(f"\n[Warning] Sampling failed: {e}\n")                   # print error message
            # [Warning] Sampling failed: VQVAE sampler is not implemented.

    def configure_optimizers(self):  

        optimizer = optim.Adam( self.model.parameters(), lr=self.params['LR'],weight_decay=self.params['weight_decay'] ) 

        gamma = self.params.get('scheduler_gamma', 0.0) 
        if gamma > 0.0: 
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)      # use scheduler
            return [optimizer], [scheduler] 
        
        return optimizer

    # print training & validation loss for each epoch
    def on_train_epoch_end(self): 
        avg_loss = torch.stack(self.training_step_outputs).mean() 
        self.log("train_epoch_loss", avg_loss, prog_bar=True, sync_dist=True) 
        print(f"\n[Epoch {self.current_epoch}] Train Loss: {avg_loss:.4f}") 
        self.train_losses.append(avg_loss.item())
        self.training_step_outputs.clear() 

    def on_validation_epoch_end(self): 
        avg_loss = torch.stack(self.validation_step_outputs).mean() 
        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True) 
        print(f"\n[Epoch {self.current_epoch}] Val Loss: {avg_loss:.4f}") 
        self.val_losses.append(avg_loss.item())
        self.validation_step_outputs.clear() 

    def on_train_end(self): 
        # import matplotlib.pyplot as plt 
        plt.figure() 
        plt.plot(self.train_losses, label="Train Loss") 
        plt.plot(self.val_losses, label="Val Loss") 
        plt.xlabel("Epoch") 
        plt.ylabel("Loss") 
        plt.legend() 
        plt.title("Training and Validation Loss") 
        plt.savefig(os.path.join(self.logger.log_dir, "loss_curve.png")) 
        plt.show() 