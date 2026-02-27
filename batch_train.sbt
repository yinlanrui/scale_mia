#!/bin/bash
#SBATCH --job-name=batch_all_train
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2    #可以改的地方
#SBATCH --output=rapid_batch-%j.log
#SBATCH --error=rapid_batch-%j.log

# Batch training script for RAPID attack framework
# This script processes multiple dataset-model combinations in sequence
# For STL-10 and GTSRB datasets with 4 different models each

echo "========================================="
echo "RAPID Batch Training Pipeline"
echo "Start time: $(date)"
echo "========================================="

# Activate conda environment
source activate rapid

# Define datasets and models to process
# To add new datasets/models, simply add to these arrays
DATASETS=("stl10" "gtsrb")  # 可以添加: "cifar10" "cifar100" "cinic" "svhn" "emnist" 等
MODELS=("vgg16" "resnet50" "densenet121" "mobilenetv2")

# Automatically build configuration file list
CONFIGS=()
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        config_file="config/${dataset}/${dataset}_${model}.json"
        if [ -f "$config_file" ]; then
            CONFIGS+=("$config_file")
        else
            echo "Warning: Configuration file not found: $config_file (skipped)"
        fi
    done
done

echo "Found ${#CONFIGS[@]} configuration files to process"

# Feature groups for attacks
FEATURE_GROUPS=("G0" "G1" "G2" "G3" "G4")

# Reference model configurations
MODEL_NUMS=(1 2 4 8 16 32 64)  # 参考模型数量
QUERY_NUMS=(1 2 4 8 16 32 64)  # 每个参考模型的查询次数

# Track success/failure
TOTAL_CONFIGS=${#CONFIGS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_CONFIGS=()

# Process each configuration
for CONFIG_FILE in "${CONFIGS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing configuration: $CONFIG_FILE"
    echo "Time: $(date)"
    echo "========================================="
    
    # Step 1: Pretrain victim and shadow models
    echo "[Step 1/3] Pretraining victim and shadow models..."
    python /public/home/yinlanrui2025/PycharmProjects/XRAPID/RAPID-main/pretrain.py 0 "$CONFIG_FILE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Pretrain failed for $CONFIG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CONFIGS+=("$CONFIG_FILE")
        continue
    fi
    echo "✓ Pretrain completed successfully"
    
    # Step 2: Train reference models
    echo "[Step 2/3] Training reference models..."
    python /public/home/yinlanrui2025/PycharmProjects/XRAPID/RAPID-main/refer_model.py "$CONFIG_FILE" --device 0
    if [ $? -ne 0 ]; then
        echo "ERROR: Reference model training failed for $CONFIG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CONFIGS+=("$CONFIG_FILE")
        continue
    fi
    echo "✓ Reference model training completed successfully"
    
    # Step 3: Run attacks for all feature groups
    echo "[Step 3/3] Running RAPID attacks for all feature groups and parameter combinations..."
    ATTACK_SUCCESS=true
    
    # 遍历所有参数组合
    for MODEL_NUM in "${MODEL_NUMS[@]}"; do
        for QUERY_NUM in "${QUERY_NUMS[@]}"; do
            for GROUP in "${FEATURE_GROUPS[@]}"; do
                echo "  Running attack: $GROUP with model_num=$MODEL_NUM, query_num=$QUERY_NUM"
                python /public/home/yinlanrui2025/PycharmProjects/XRAPID/RAPID-main/mia_attack.py 0 "$CONFIG_FILE" \
                    --feature_group "$GROUP" \
                    --model_num "$MODEL_NUM" \
                    --query_num "$QUERY_NUM"
                if [ $? -ne 0 ]; then
                    echo "  ERROR: Attack with $GROUP (r${MODEL_NUM}_q${QUERY_NUM}) failed for $CONFIG_FILE"
                    ATTACK_SUCCESS=false
                    break 3  # 跳出所有三层循环
                fi
                echo "  ✓ Attack with $GROUP (r${MODEL_NUM}_q${QUERY_NUM}) completed"
            done
        done
    done
    
    if [ "$ATTACK_SUCCESS" = true ]; then
        echo "✓ All attacks completed successfully for $CONFIG_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "ERROR: Some attacks failed for $CONFIG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CONFIGS+=("$CONFIG_FILE")
    fi
    
    echo "Completed processing: $CONFIG_FILE"
done

# Final summary
echo ""
echo "========================================="
echo "RAPID Batch Training Complete"
echo "End time: $(date)"
echo "========================================="
echo "Total configurations: $TOTAL_CONFIGS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed configurations:"
    for CONFIG in "${FAILED_CONFIGS[@]}"; do
        echo "  - $CONFIG"
    done
fi

echo ""
echo "You can now generate analysis logs using:"
echo "python generate_attack_logs.py --all"
echo "========================================="
