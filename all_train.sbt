#!/bin/bash
#SBATCH --job-name=rapid_pipeline
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2  
 
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)
OUTPUT_FILE="rapid_pipeline-${JOB_ID}-${NODE_NAME}.log"
 
echo "======================================================================" >> $OUTPUT_FILE
echo "RAPID Attack Pipeline Started at: $(date)" >> $OUTPUT_FILE
echo "Running on node: $NODE_NAME" >> $OUTPUT_FILE
echo "Using GPUs:" >> $OUTPUT_FILE
nvidia-smi >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
 
source activate v100  

# 配置文件路径
CONFIG_FILE="config/stl10/stl10_vgg16.json"
WORK_DIR="/public/home/yinlanrui2025/PycharmProjects/RAPID-main"

# 特征组合列表
FEATURE_GROUPS=("G0" "G1" "G2" "G3" "G4")

echo "" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
echo "Step 1: Training Victim and Shadow Models" >> $OUTPUT_FILE
echo "Config: $CONFIG_FILE" >> $OUTPUT_FILE
echo "Started at: $(date)" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
srun python ${WORK_DIR}/pretrain.py 0 $CONFIG_FILE >> $OUTPUT_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Pretrain failed!" >> $OUTPUT_FILE
    exit 1
fi
echo "✓ Pretrain completed at: $(date)" >> $OUTPUT_FILE

echo "" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
echo "Step 2: Training Reference Models" >> $OUTPUT_FILE
echo "Started at: $(date)" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
srun python ${WORK_DIR}/refer_model.py 0 $CONFIG_FILE >> $OUTPUT_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Reference model training failed!" >> $OUTPUT_FILE
    exit 1
fi
echo "✓ Reference models completed at: $(date)" >> $OUTPUT_FILE

echo "" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
echo "Step 3: Executing RAPID Attacks with Different Feature Groups" >> $OUTPUT_FILE
echo "Started at: $(date)" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE

for FEATURE_GROUP in "${FEATURE_GROUPS[@]}"; do
    echo "" >> $OUTPUT_FILE
    echo "----------------------------------------------------------------------" >> $OUTPUT_FILE
    echo "Attacking with Feature Group: $FEATURE_GROUP" >> $OUTPUT_FILE
    echo "Started at: $(date)" >> $OUTPUT_FILE
    echo "----------------------------------------------------------------------" >> $OUTPUT_FILE
    srun python ${WORK_DIR}/mia_attack.py 0 $CONFIG_FILE --feature_group $FEATURE_GROUP >> $OUTPUT_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "WARNING: Attack with $FEATURE_GROUP failed!" >> $OUTPUT_FILE
    else
        echo "✓ Attack with $FEATURE_GROUP completed at: $(date)" >> $OUTPUT_FILE
    fi
done

echo "" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE
echo "All Tasks Completed!" >> $OUTPUT_FILE
echo "Job ended at: $(date)" >> $OUTPUT_FILE
echo "======================================================================" >> $OUTPUT_FILE