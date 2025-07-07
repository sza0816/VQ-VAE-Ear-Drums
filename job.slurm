#!/bin/bash 
#SBATCH --job-name=vqvae         #  
#SBATCH --output=logs/vqvae.out  #  
#SBATCH --error=logs/vqvae.err 
#SBATCH --nodes=1                #  
#SBATCH --ntasks=1               #  
#SBATCH --cpus-per-task=4        #  
#SBATCH --gres=gpu:1             #  
#SBATCH --time=04:00:00          #  
#SBATCH --partition=ciaq          #  
#SBATCH --mem=16G

echo "Job started on $(date)" 
echo "Running on node: $(hostname)"  

python run.py -c configs/vq_vae.yaml 

echo "Job finished on $(date)" 