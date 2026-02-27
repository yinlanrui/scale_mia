#!/bin/bash
#SBATCH --job-name=batch_cross_arch
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090:2
#SBATCH --output=cross_arch_batch-%j.log
#SBATCH --error=cross_arch_batch-%j.log

# Batch cross-architecture attack script for RAPID framework
# This script runs cross-architecture MIA attacks with different model combinations
# Testing architecture transfer robustness

echo "========================================="
echo "RAPID Batch Cross-Architecture Attack Pipeline"
echo "Start time: $(date)"
echo "========================================="

# Activate conda environment
source activate v100

# Define datasets to test
DATASETS=("cifar10")

# Define model architectures
MODELS=("vgg16" "resnet50" "densenet121" "mobilenetv2")

# Feature groups for attacks
FEATURE_GROUPS=("G0" "G4")

# Reference model configurations
MODEL_NUMS=(4)  # 参考模型数量
QUERY_NUMS=(8)    # 每个参考模型的查询次数

# Track success/failure
TOTAL_ATTACKS=0
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_ATTACKS=()

# Process each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing dataset: $DATASET"
    echo "Time: $(date)"
    echo "========================================="
    
    # Process each victim architecture
    for VICTIM_ARCH in "${MODELS[@]}"; do
        # Process each shadow/reference architecture
        for SHADOW_REF_ARCH in "${MODELS[@]}"; do
            
            echo ""
            echo "-----------------------------------------"
            echo "Scenario: Victim=$VICTIM_ARCH, Shadow/Ref=$SHADOW_REF_ARCH"
            echo "-----------------------------------------"
            
            # Build config file path based on victim architecture
            CONFIG_FILE="config/${DATASET}/${DATASET}_${VICTIM_ARCH}.json"
            
            # Check if config file exists
            if [ ! -f "$CONFIG_FILE" ]; then
                echo "Warning: Configuration file not found: $CONFIG_FILE (skipped)"
                continue
            fi
            
            # Check if victim model exists
            VICTIM_MODEL_PATH="results/${DATASET}_${VICTIM_ARCH}/victim_model/best.pth"
            if [ ! -f "$VICTIM_MODEL_PATH" ]; then
                echo "Warning: Victim model not found: $VICTIM_MODEL_PATH (skipped)"
                continue
            fi
            
            # Check if shadow model exists
            SHADOW_MODEL_PATH="results/${DATASET}_${SHADOW_REF_ARCH}/shadow_model/best.pth"
            if [ ! -f "$SHADOW_MODEL_PATH" ]; then
                echo "Warning: Shadow model not found: $SHADOW_MODEL_PATH (skipped)"
                continue
            fi
            
            # Check if reference models exist
            REF_MODEL_PATH="results/${DATASET}_${SHADOW_REF_ARCH}/RAPID/reference_model_0/best.pth"
            if [ ! -f "$REF_MODEL_PATH" ]; then
                echo "Warning: Reference models not found for ${DATASET}_${SHADOW_REF_ARCH} (skipped)"
                continue
            fi
            
            # Run attacks for all parameter combinations
            SCENARIO_SUCCESS=true
            
            for MODEL_NUM in "${MODEL_NUMS[@]}"; do
                for QUERY_NUM in "${QUERY_NUMS[@]}"; do
                    for GROUP in "${FEATURE_GROUPS[@]}"; do
                        TOTAL_ATTACKS=$((TOTAL_ATTACKS + 1))
                        
                        echo "  Running attack: $GROUP (r${MODEL_NUM}_q${QUERY_NUM})"
                        echo "    Victim: $VICTIM_ARCH | Shadow/Ref: $SHADOW_REF_ARCH"
                        
                        python /public/home/yinlanrui2025/PycharmProjects/XRAPID/RAPID-main/mia_attack_cross_arch.py 0 "$CONFIG_FILE" \
                            --dataset_name "$DATASET" \
                            --victim_model_name "$VICTIM_ARCH" \
                            --shadow_model_name "$SHADOW_REF_ARCH" \
                            --reference_model_name "$SHADOW_REF_ARCH" \
                            --feature_group "$GROUP" \
                            --model_num "$MODEL_NUM" \
                            --query_num "$QUERY_NUM"
                        
                        if [ $? -ne 0 ]; then
                            echo "  ERROR: Attack failed"
                            FAIL_COUNT=$((FAIL_COUNT + 1))
                            FAILED_ATTACKS+=("${DATASET} | V:${VICTIM_ARCH} S/R:${SHADOW_REF_ARCH} | ${GROUP} r${MODEL_NUM}_q${QUERY_NUM}")
                            SCENARIO_SUCCESS=false
                            break 3
                        fi
                        
                        echo "  ✓ Attack completed"
                        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                    done
                done
            done
            
            if [ "$SCENARIO_SUCCESS" = true ]; then
                echo "✓ All attacks completed for scenario: V:$VICTIM_ARCH S/R:$SHADOW_REF_ARCH"
            else
                echo "ERROR: Some attacks failed for scenario: V:$VICTIM_ARCH S/R:$SHADOW_REF_ARCH"
            fi
        done
    done
    
    echo "Completed dataset: $DATASET"
done

# Final summary
echo ""
echo "========================================="
echo "RAPID Batch Cross-Architecture Attack Complete"
echo "End time: $(date)"
echo "========================================="
echo "Total attacks attempted: $TOTAL_ATTACKS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed attacks:"
    for ATTACK in "${FAILED_ATTACKS[@]}"; do
        echo "  - $ATTACK"
    done
fi

echo ""
echo "Results are saved in: results/cross_arch_victim_*_shadow_*_ref_*/"
echo "========================================="
