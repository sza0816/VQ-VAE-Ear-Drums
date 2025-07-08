import os
import math
import torch
from torch import optim 
from models import BaseVAE 
from models.types_ import * 
import pytorch_lightning as pl 
from torchvision import utils as vutils 
import matplotlib.pyplot as plt 

from torchmetrics.image import PeakSignalNoiseRatio 
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from pytorch_lightning.utilities.rank_zero import rank_zero_only 

class VAEXperiment(pl.LightningModule): 
    def __init__(self, vae_model: BaseVAE, params: dict) -> None: 
        super(VAEXperiment, self).__init__() 

        self.model = vae_model 
        self.params = params 
        self.curr_device = None 
    
        self.training_step_outputs = [] 
        self.validation_step_outputs = [] 
        self.train_losses = [] 
        self.val_losses = [] 
    
        self.psnr = PeakSignalNoiseRatio() 
        self.ssim = StructuralSimilarityIndexMeasure() 
 
        self.val_psnrs = [] 
        self.val_ssims = [] 
    
        self.validation_recons = [] 
 
    def forward(self, input: Tensor, **kwargs) -> Tensor: 
        return self.model(input, **kwargs) 

    def training_step(self, batch, batch_idx): 
        real_img, labels = batch 
        self.curr_device = real_img.device 
 
        results = self.forward(real_img, labels=labels) 
        train_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'], batch_idx=batch_idx) 

        self.training_step_outputs.append(train_loss['loss']) 
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True) 
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx): 
        real_img, labels = batch 
        self.curr_device = real_img.device 
 
        results = self.forward(real_img, labels=labels) 
        val_loss = self.model.loss_function(*results, M_N=1.0, batch_idx=batch_idx) 
 
        self.validation_step_outputs.append(val_loss['loss']) 
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True) 
 
        recon = results[0].detach().cpu() 
        self.validation_recons.append((real_img.cpu(), recon)) 
 
        return val_loss['loss'] 

    def on_validation_end(self) -> None: 
        self.sample_images() 
 
    def sample_images(self): 
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader())) 
        test_input = test_input.to(self.curr_device) 
        test_label = test_label.to(self.curr_device) 
 
        recons = self.model.generate(test_input, labels=test_label) 
        vutils.save_image(recons.data, 
                        os.path.join(self.logger.log_dir, 
                                    "Reconstructions", 
                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"), 
                        normalize=True, 
                        nrow=8) 

        try: 
            samples = self.model.sample(144, self.curr_device, labels=test_label) 
            vutils.save_image(samples.cpu().data, 
                            os.path.join(self.logger.log_dir, 
                                        "Samples",
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"), 
                                        normalize=True, 
                                        nrow=12) 
        except Exception as e: 
            print(f"\n[Warning] Sampling failed: {e}\n")           ### print sample err msg

    def configure_optimizers(self): 
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay']) 
    
        gamma = self.params.get('scheduler_gamma', 0.0) 
        if gamma > 0.0: 
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) 
            return [optimizer], [scheduler] 
    
        return optimizer
    
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

        # calculate PSNR, SSIM 
        all_psnr = [] 
        all_ssim = [] 
        for real, recon in self.validation_recons: 
            all_psnr.append(self.psnr(recon, real)) 
            all_ssim.append(self.ssim(recon, real)) 
    
        avg_psnr = torch.stack(all_psnr).mean() 
        avg_ssim = torch.stack(all_ssim).mean() 
 
        self.val_psnrs.append(avg_psnr.item()) 
        self.val_ssims.append(avg_ssim.item()) 
    
        self.log("val_psnr", avg_psnr, prog_bar=True, sync_dist=True) 
        self.log("val_ssim", avg_ssim, prog_bar=True, sync_dist=True) 
    
        print(f"\n[Epoch {self.current_epoch}] Val PSNR: {avg_psnr:.2f}, Val SSIM: {avg_ssim:.4f}") 
    
        self.validation_recons.clear()
    
    @rank_zero_only 
    def on_train_end(self): 
        # training validation loss plot 
        plt.figure() 
        plt.plot(self.train_losses, label="Train Loss") 
        plt.plot(self.val_losses, label="Val Loss") 
        plt.xlabel("Epoch") 
        plt.ylabel("Loss") 
        plt.legend() 
        plt.title("Training and Validation Loss") 
        plt.savefig(os.path.join(self.logger.log_dir, "loss_curve.png")) 
        plt.show() 
    
        # psnr, ssim plot 
        plt.figure() 
        plt.plot(self.val_psnrs, label="Val PSNR") 
        plt.plot(self.val_ssims, label="Val SSIM") 
        plt.xlabel("Epoch") 
        plt.ylabel("Metric") 
        plt.legend() 
        plt.title("Validation Metrics") 
        plt.savefig(os.path.join(self.logger.log_dir, "val_metrics.png")) 
        plt.show() 
    