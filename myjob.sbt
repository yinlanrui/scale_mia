#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2  
 
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)
OUTPUT_FILE="my_gpu_job-${JOB_ID}-${NODE_NAME}.log"
 
echo "Job started at: $(date)" >> $OUTPUT_FILE
echo "Running on node: $NODE_NAME" >> $OUTPUT_FILE
echo "Using GPUs:" >> $OUTPUT_FILE
nvidia-smi >> $OUTPUT_FILE
 
source activate v100  

srun python /public/home/yinlanrui2025/PycharmProjects/RAPID-main/plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack >> $OUTPUT_FILE 2>&1  

echo "Job ended at: $(date)" >> $OUTPUT_FILE