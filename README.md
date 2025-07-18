<h1 align="center">
  <b>VQ-VAE for Ear Drum Reconstructions</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.5-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.3-2BAF2B.svg" /></a>
       <a href= "https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
         <a href= "https://twitter.com/intent/tweet?text=PyTorch-VAE:%20Collection%20of%20VAE%20models%20in%20PyTorch.&url=https://github.com/AntixK/PyTorch-VAE">
        <img src="https://img.shields.io/twitter/url/https/shields.io.svg?style=social" /></a>

</p>

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
# squeue -u <username>     # check job status

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

$ This repository adapts AntixK's code for a specific medical imaging use case.
```