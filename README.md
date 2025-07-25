<h1 align="center">
  <b>VQ-VAE for Ear Drum Reconstructions</b><br>
</h1>


##### This Repository is adapted from https://github.com/AntixK/PyTorch-VAE. Appreciate it!

**Update 7/3/2025:** Adjusted PyTorch Lightning import commands to support 2.5.2 version

This project reproduces the VQ-VAE implementation by AntixK, adapted to reconstruct video frames of eardrums as part of a larger medical imaging project. 

Note: the eardrum datasets are not included in this repository due to privacy constraints. A job.slurm file is provided to run the model on an HPC cluster. 

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 (tested up to 2.5.2)
- All experiments were run on a CUDA-enabled GPU

### Installation
```
$ git clone git@github.com:sza0816/VQ-VAE-Ear-Drums.git
$ cd Ear-Drum-VQVAE
$ pip install -r requirements.txt        # --no-user if in virtual env
```

### Usage
```
$ conda activate <env>
$ cd Ear-Drum-VQVAE           # if needed
$ sbatch job.slurm
# squeue -u <username>        # check job status

# or run locally
$python run.py --conf configs/vq_vae.yaml
```
**Config file template - See folder "configs"**

```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 3
  latent_dim: 
    .         # Other parameters required by the model
    .
    .

data_params:
  data_path: "<path to the dataset>"
  train_batch_size: 64 # Better to have a square number
  val_batch_size:  64
  patch_size: 64  # Models are designed to work for this size
  num_workers: 4
  
exp_params:
  manual_seed: 1265
  LR: 0.005
  weight_decay:
    .         # Other arguments required for training, like scheduler etc.
    .
    .

trainer_params:
  gpus: 1         
  max_epochs: 100
  gradient_clip_val: 1.5
    .
    .
    .

logging_params:
  save_dir: "logs/"
  name: "<experiment name>"
```

**View TensorBoard Logs**
```
$ cd logs/<experiment name>/version_<the version you want>
$ tensorboard --logdir .
```
### Known Warnings
You may encounter warnings related to deprecated pretrained parameters in ```torchvision```

```
UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated...
```

They are triggered internally by LPIPS. I chose not to modify the library code since they do not affect the evaluation. You can savely ignore them.

### Output Structure
Each training run (via sbatch job.slurm) creates a new folder under ```logs/VQVAE/```

```
logs/ 
└── VQVAE/ 
    ├── version_n/ 
    │   ├── checkpoints/                      # Model checkpoints 
    │   ├── Reconstructions/                  # Reconstructed images per epoch 
    │   ├── codebook_usage_cdf.png            # Codebook usage CDF 
    │   ├── fid_lpips_curve.png               # FID and LPIPS curves 
    │   ├── loss_curve.png                    # Training loss curve 
    │   ├── mse_nlpd_curve.png                # MSE and NLPD curves 
    │   ├── psnr_ssim_msssim_curves.png       # PSNR, SSIM, MS-SSIM curves 
    │   ├── final_metrics.txt                 # Metrics from the last epoch 
    │   ├── hparams.yaml                      # Hyperparameters for this run 
    │   └── events.out.tfevents...            # TensorBoard log file 
    ├── vqvae.out                             # Stdout log (epoch metrics, progress) 
    └── vqvae.err                             # Stderr log (warnings, device info) 
```

 - ```vqvae.out``` logs the full training process, including metric values per epoch. 
 - ```vqvae.err``` contains environment info and expected warnings (LPIPS-related deprecations)

### Evaluation
Several variants of the straight-through estimator (STE) for the quantized output q (quantized_latents) in the VQ-VAE forward pass were experimented. The results are summarized below: 

 - **Base case** (standard STE):
   - formula: `q = e + (q - z).detach() `
   - Reconstructions: slightly blurry
   - Reference: version_7
 - **Modified STE 1**: 
   - formula: `q = e + q - z.detach()`
   - Reconstructions: noticeably blurrier than base
   - Reference: version_8
 - **Modified STE 2** (misunderstood version):
   - formula: `q = e + alpha * (q - z).detach()`
   - Reconstructions: significantly sharper
   - **Note**: The formula performs well but lacks theoretical justification
   - Reference: version_9
 - **Modified STE 3** (intended alternative):
   - formula: `q = e + alpha * q + (beta * q - z).detach()`
   - Reconstructions: slightly blurrier than base
   - **Note**: May require more tuning; not further optimized due to time
   - Reference: version_10 to version_13

**Final Reconstructions Comparison**
| Variant     | Formula                                  | Codebook Usage | Final LPIPS | Sample Reconstructions                 |
|-------------|------------------------------------------|----------------|-------------|----------------------------------------|
| **Base**    | `q = e + (q - z).detach()`               | ~70%           | ~0.10       | ![](examples/version_7_epoch_216.png)  |
| **STE #1**  | `q = e + q - z.detach()`                 | 30–40%         | 0.10–0.20   | ![](examples/version_8_epoch_194.png)  |
| **STE #2**  | `q = e + α * (q - z).detach()`           | 15–30%         | ~0.05       | ![](examples/version_9_epoch_217.png)  |
| **STE #3**  | `q = e + α * q + (β * q - z).detach()`   | 15–50%         | 0.13–0.18   | ![](examples/version_12_epoch_224.png) |

### License
**Apache License 2.0**

| Permissions      | Limitations       | Conditions                       |
|------------------|-------------------|----------------------------------|
| ✔️ Commercial use |  ❌  Trademark use |  ⓘ License and copyright notice | 
| ✔️ Modification   |  ❌  Liability     |  ⓘ State changes                |
| ✔️ Distribution   |  ❌  Warranty      |                                  |
| ✔️ Patent use     |                   |                                  |
| ✔️ Private use    |                   |                                  |


### Citation
```
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}

# This repository adapts AntixK's code for a specific medical imaging use case.
```