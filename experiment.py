import os
import math
import torch
from torch import optim 
from models import BaseVAE 
from models.types_ import * 
import pytorch_lightning as pl 
from torchvision import utils as vutils 
import matplotlib.pyplot as plt 

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim_fn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import lpips

class VAEXperiment(pl.LightningModule): 
    def __init__(self, vae_model: BaseVAE, params: dict) -> None: 
        super(VAEXperiment, self).__init__() 

        self.model = vae_model 
        self.params = params 
        self.curr_device = None
        self.register_buffer("codebook_usage", torch.zeros(self.model.num_embeddings, dtype=torch.long))
    
        self.training_step_outputs = [] 
        self.validation_step_outputs = [] 
        self.train_losses = [] 
        self.val_losses = [] 
    
        self.psnr = PeakSignalNoiseRatio() 
        self.ssim = StructuralSimilarityIndexMeasure() 
 
        self.train_psnrs=[]              # train psnr
        self.train_ssims=[]              # train ssim

        self.val_psnrs = []              # validation psnr
        self.val_ssims = []              # validation ssim
    
        self.validation_recons = []

        self.train_mses = []             # MSE
        self.val_mses = []
        self.mse_loss_fn = torch.nn.MSELoss()

        self.fid_metric = FrechetInceptionDistance(feature = 2048, reset_real_features = False)         # fid
        self.val_fids = []

        self.train_ms_ssims = []                           # ms-ssim
        self.val_ms_ssims = []
 
        self.val_nlpds = []                              # nlpd

        self.lpips_metrics = lpips.LPIPS(net='alex').to(self.curr_device if self.curr_device else 'cuda')       # lpips
        self.val_lpips = []

    def forward(self, input: Tensor, **kwargs) -> Tensor: 
        return self.model(input, **kwargs) 

    def training_step(self, batch, batch_idx):                                # 
        real_img, labels = batch 
        self.curr_device = real_img.device 
 
        # results = self.forward(real_img, labels=labels) 
        # train_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'], batch_idx=batch_idx) 
        recons, input_img, vq_loss, encoding_inds = self.forward(real_img, labels=labels)  
        # recons, input_img, vq_loss, encoding_inds = results 
        flat_inds = encoding_inds.flatten()  
        self.codebook_usage.scatter_add_(0, flat_inds, torch.ones_like(flat_inds, dtype=self.codebook_usage.dtype)) 
        train_loss = self.model.loss_function(recons, input_img, vq_loss, M_N=self.params['kld_weight'], batch_idx=batch_idx) 

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
        self.validation_recons.append((real_img, recon)) 
 
        return val_loss['loss'] 

    def on_validation_end(self) -> None: 
        self.sample_images() 
 
    def sample_images(self): 
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader())) 
        test_input = test_input.to(self.curr_device) 
        test_label = test_label.to(self.curr_device) 
 
        recons = self.model.generate(test_input, labels=test_label)                      # image reconstruction
        vutils.save_image(recons.data, 
                        os.path.join(self.logger.log_dir, 
                                    "Reconstructions", 
                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"), 
                        normalize=True, 
                        nrow=8) 

        # try:                                                                             # image sampling, not implemented
        #     samples = self.model.sample(144, self.curr_device, labels=test_label) 
        #         vutils.save_image(samples.cpu().data, 
        #                         os.path.join(self.logger.log_dir, 
        #                                     "Samples",
        #                                     f"{self.logger.name}_Epoch_{self.current_epoch}.png"), 
        #                                     normalize=True, 
        #                                     nrow=12) 
        # except Exception as e: 
        #     # print(f"\n[Warning] Sampling failed: {e}\n")
        #     pass

    def configure_optimizers(self): 
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay']) 
    
        gamma = self.params.get('scheduler_gamma', 0.0) 
        if gamma > 0.0: 
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) 
            return [optimizer], [scheduler] 
    
        return optimizer
    
    def on_train_epoch_end(self):                                                # 
        avg_loss = torch.stack(self.training_step_outputs).mean() 
        self.log("train_epoch_loss", avg_loss, prog_bar=True, sync_dist=True) 
        print(f"\n[Epoch {self.current_epoch}] Train Loss: {avg_loss:.4f}") 
        self.train_losses.append(avg_loss.item()) 
        self.training_step_outputs.clear()

        # training psnr & ssim
        all_train_psnr = [] 
        all_train_ssim = [] 
        
        train_loader = self.trainer.datamodule.train_dataloader()         # run train loader after this epoch

        all_train_mse = []
        all_train_ms_ssim = []
 
        for real_img, labels in train_loader: 
            real_img = real_img.to(self.curr_device) 
            labels = labels.to(self.curr_device) 
        
            with torch.no_grad(): 
                # take reconstruction result 
                recons = self.model(real_img, labels=labels)[0] 
        
            all_train_psnr.append(self.psnr(recons, real_img)) 
            all_train_ssim.append(self.ssim(recons, real_img)) 
            all_train_mse.append(self.mse_loss_fn(recons, real_img))
            all_train_ms_ssim.append(msssim_fn(
                preds=recons,
                target=real_img, 
                kernel_size=3,
                betas=(0.4,0.4,0.2),
                data_range=1.0
            ))
 
        # avg metric values of each batch
        avg_train_psnr = torch.stack(all_train_psnr).mean() 
        avg_train_ssim = torch.stack(all_train_ssim).mean() 
        avg_train_mse = torch.stack(all_train_mse).mean()
        avg_train_ms_ssim = torch.stack(all_train_ms_ssim).mean()
        
        self.log("train_psnr", avg_train_psnr, prog_bar=True, sync_dist=True)  
        self.log("train_ssim", avg_train_ssim, prog_bar=True, sync_dist=True)
        self.log("train_mse", avg_train_mse, prog_bar=True)
        self.log("train_ms_ssim", avg_train_ms_ssim, prog_bar=True, sync_dist=True)
        
        # print metrics
        print(
            f"\n[Epoch {self.current_epoch}]"
            f"\nTrain PSNR: {avg_train_psnr:.2f},"
            f"\nTrain SSIM: {avg_train_ssim:.4f},"
            f"\nTrain MSE: {avg_train_mse:.6f}, "
            f"\nTrain MS-SSIM: {avg_train_ms_ssim}"
            )  
        
        # append metrics to list
        self.train_psnrs.append(avg_train_psnr.item()) 
        self.train_ssims.append(avg_train_ssim.item()) 
        self.train_mses.append(avg_train_mse.item())
        self.train_ms_ssims.append(avg_train_ms_ssim.item())
 
    def on_validation_epoch_end(self):                                                # 
        avg_loss = torch.stack(self.validation_step_outputs).mean() 
        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True) 
        print(f"\n[Epoch {self.current_epoch}] Val Loss: {avg_loss:.4f}") 
        self.val_losses.append(avg_loss.item()) 
        self.validation_step_outputs.clear() 

        # PSNR, SSIM, MSE, FID, MS-SSIM, NLPD
        all_psnr = [] 
        all_ssim = [] 
        all_mse = []
        all_ms_ssim = []
        all_nlpd = []
        all_lpips = []

        for real, recon in self.validation_recons: 
            real = real.to(self.curr_device)
            recon = recon.to(self.curr_device)

            all_psnr.append(self.psnr(recon, real)) 
            all_ssim.append(self.ssim(recon, real)) 
            all_mse.append(self.mse_loss_fn(recon, real))
            all_ms_ssim.append(msssim_fn(
                preds=recon, 
                target=real, 
                kernel_size=3, 
                betas=(0.4,0.4,0.2),
                data_range=1.0
            ))

            real_255 = (real * 255).clamp(0, 255).to(torch.uint8).to(self.curr_device)          # fid
            recon_255 = (recon * 255).clamp(0, 255).to(torch.uint8).to(self.curr_device)
            self.fid_metric.update(real_255, real=True)
            self.fid_metric.update(recon_255, real=False)

            sigma2_tensor = torch.tensor(1.0, device=self.curr_device)
            const_term = 0.5 * torch.log(2 * math.pi * sigma2_tensor)

            pixel_mse = (real-recon)**2                                                        # nlpd
            nlpd_map = const_term + pixel_mse/(2*sigma2_tensor)
            nlpd_sample = nlpd_map.mean()
            all_nlpd.append(nlpd_sample.cpu())

            with torch.no_grad():
                lpips_value = self.lpips_metrics(recon, real).mean()
            all_lpips.append(lpips_value)

        # average
        avg_psnr = torch.stack(all_psnr).mean() 
        avg_ssim = torch.stack(all_ssim).mean()
        avg_mse = torch.stack(all_mse).mean()
        fid_score = self.fid_metric.compute()
        avg_ms_ssim = torch.stack(all_ms_ssim).mean()
        avg_nlpd = torch.stack(all_nlpd).mean()
        avg_lpips = torch.stack(all_lpips).mean()

        # append avg metrics to list
        self.val_psnrs.append(avg_psnr.item()) 
        self.val_ssims.append(avg_ssim.item()) 
        self.val_mses.append(avg_mse.item())
        self.val_ms_ssims.append(avg_ms_ssim.item())

        self.fid_metric.reset()                         # fid append to list
        self.val_fids.append(fid_score.item())

        self.val_nlpds.append(avg_nlpd.item())
        self.val_lpips.append(avg_lpips.item())
    
        # log
        self.log("val_psnr", avg_psnr, prog_bar=True, sync_dist=True) 
        self.log("val_ssim", avg_ssim, prog_bar=True, sync_dist=True) 
        self.log("val_mse", avg_mse, prog_bar=True)
        self.log("val_fid", fid_score, prog_bar=True, sync_dist=True)
        self.log("val_ms_ssim", avg_ms_ssim, prog_bar=True, sync_dist=True)
        self.log("val_nlpd", avg_nlpd, prog_bar=True, sync_dist=True)
        self.log("val_lpips", avg_lpips, prog_bar=True, sync_dist=True)

        print(
            f"\n[Epoch {self.current_epoch}]"
            f"\nVal PSNR: {avg_psnr:.2f},"
            f"\nVal SSIM: {avg_ssim:.4f},"
            f"\nVal MSE: {avg_mse:.6f},"
            f"\nVal FID: {fid_score:.2f},"
            f"\nVal MS-SSIM: {avg_ms_ssim:.4f},"
            f"\nVal NLPD: {avg_nlpd:.4f},"
            f"\nVal LPIPS: {avg_lpips:.4f}") 
    
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
        plt.close() 

        fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 

        metrics = [ 
            ("PSNR", self.train_psnrs, self.val_psnrs), 
            ("SSIM", self.train_ssims, self.val_ssims), 
            ("MS-SSIM", self.train_ms_ssims, self.val_ms_ssims)
        ] 
        for ax, (metric_name, train_values, val_values) in zip(axs, metrics): 
            ax.plot(train_values, label=f"Train {metric_name}", linestyle='-') 
            ax.plot(val_values, label=f"Val {metric_name}", linestyle='--') 
            ax.set_xlabel("Epoch") 
            ax.set_ylabel(metric_name) 
            ax.set_title(f"{metric_name} Metrics") 
            ax.legend() 
            ax.grid(True) 
        plt.tight_layout() 
        plt.savefig(os.path.join(self.logger.log_dir, "psnr_ssim_msssim_curves.png")) 
        plt.close() 

        # mse, nlpd
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

        axs[0].plot(self.train_mses, label="Train MSE") 
        axs[0].plot(self.val_mses, label="Validation MSE") 
        axs[0].axhline(0, color='grey', linestyle='--', label='Reference: MSE=0') 
        axs[0].set_xlabel("Epoch") 
        axs[0].set_ylabel("MSE") 
        axs[0].set_title("Pixel Level MSE") 
        axs[0].legend()
        axs[0].grid(True) 

        axs[1].plot(self.val_nlpds, label="Val NLPD") 
        axs[1].set_xlabel("Epoch") 
        axs[1].set_ylabel("NLPD") 
        axs[1].set_title("Validation NLPD") 
        axs[1].legend()
        axs[1].grid(True) 

        plt.tight_layout() 
        plt.savefig(os.path.join(self.logger.log_dir, "mse_nlpd_curve.png")) 
        plt.close() 

        # fid, lpips
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

        axs[0].plot(self.val_fids, label="Val FID") 
        axs[0].set_xlabel("Epoch") 
        axs[0].set_ylabel("FID") 
        axs[0].set_title("Validation FID") 
        axs[0].legend() 
        axs[0].grid(True) 

        axs[1].plot(self.val_lpips, label="Val LPIPS") 
        axs[1].set_xlabel("Epoch") 
        axs[1].set_ylabel("LPIPS") 
        axs[1].set_title("Validation LPIPS") 
        axs[1].legend() 
        axs[1].grid(True) 

        plt.tight_layout() 
        plt.savefig(os.path.join(self.logger.log_dir, "fid_lpips_curve.png")) 
        plt.close()



        # print metrics of the final epoch
        print("\n[Final Epoch Metrics]") 
        print(f"{'Metric':<15} {'Train':>12} {'Validation':>12}") 
        print("-" * 41) 
        print(f"{'Loss':<15} {self.train_losses[-1]:>12.6f} {self.val_losses[-1]:>12.6f}") 
        print(f"{'PSNR':<15} {self.train_psnrs[-1]:>12.2f} {self.val_psnrs[-1]:>12.2f}") 
        print(f"{'SSIM':<15} {self.train_ssims[-1]:>12.4f} {self.val_ssims[-1]:>12.4f}") 
        print(f"{'MSE':<15} {self.train_mses[-1]:>12.6f} {self.val_mses[-1]:>12.6f}") 
        print(f"{'FID':<15} {'-':>12} {self.val_fids[-1]:>12.4f}")
        print(f"{'MS-SSIM':<15} {self.train_ms_ssims[-1]:>12.4f} {self.val_ms_ssims[-1]:>12.4f}")
        print(f"{'NLPD':<15} {'-':>12} {self.val_nlpds[-1]:>12.4f}")
        print(f"{'LPIPS':<15} {'-':>12} {self.val_lpips[-1]:>12.4f}") 