#!/bin/bash
#SBATCH --job-name=plot_feature_comparisons
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
 
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)

# 主日志文件
MAIN_LOG="logs/plot_feature_comparisons_${JOB_ID}.log"
mkdir -p logs

echo "======================================" > $MAIN_LOG
echo "Feature Groups Comparison Job started at: $(date)" >> $MAIN_LOG
echo "Running on node: $NODE_NAME" >> $MAIN_LOG
echo "Using GPUs:" >> $MAIN_LOG
nvidia-smi >> $MAIN_LOG
echo "======================================" >> $MAIN_LOG
 
source activate v100

# 定义所有组合
DATASETS=("fashion_mnist")
MODELS=("vgg16" "resnet50" "mobilenetv2" "densenet121")
# ATTACK_MODEL="mia_fc"
FEATURE_GROUPS=("G0" "G1" "G2" "G3" "G4")  # 特征组配置数组

# 统计信息
TOTAL_TASKS=$((${#DATASETS[@]} * ${#MODELS[@]}))
CURRENT_TASK=0
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "" >> $MAIN_LOG
echo "Total tasks to execute: $TOTAL_TASKS" >> $MAIN_LOG
echo "Datasets: ${DATASETS[@]}" >> $MAIN_LOG
echo "Models: ${MODELS[@]}" >> $MAIN_LOG
# echo "Attack Model: $ATTACK_MODEL" >> $MAIN_LOG
echo "Feature Groups: ${FEATURE_GROUPS[@]} (对比)" >> $MAIN_LOG
echo "" >> $MAIN_LOG

# 循环所有数据集和模型组合
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        
        echo "========================================" >> $MAIN_LOG
        echo "Task $CURRENT_TASK/$TOTAL_TASKS: $dataset + $model (${FEATURE_GROUPS[@]} comparison)" >> $MAIN_LOG
        echo "Started at: $(date)" >> $MAIN_LOG
        echo "========================================" >> $MAIN_LOG
        
        # 动态构建攻击列表
        ATTACKS=""
        for fg in "${FEATURE_GROUPS[@]}"; do
            if [ -z "$ATTACKS" ]; then
                # ATTACKS="rapid_attack_${fg}_${ATTACK_MODEL}"
                ATTACKS="rapid_attack_${fg}"
            else
                # ATTACKS="${ATTACKS},rapid_attack_${fg}_${ATTACK_MODEL}"
                ATTACKS="${ATTACKS},rapid_attack_${fg}"
            fi
        done
        
        # 临时日志
        TEMP_LOG="/tmp/plot_${dataset}_${model}_features_${JOB_ID}.log"
        
        # 运行绘图脚本
        srun python /public/home/yinlanrui2025/PycharmProjects/XRAPID/RAPID-main/plot.py 0 \
            config/${dataset}/${dataset}_${model}.json \
            --dataset_name $dataset \
            --model_name $model \
            --attacks $ATTACKS > $TEMP_LOG 2>&1
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ SUCCESS: $dataset + $model (${FEATURE_GROUPS[@]} features)" >> $MAIN_LOG
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            
            # 提取保存目录并移动日志
            SAVE_DIR=$(grep "LOG_SAVE_DIR=" $TEMP_LOG | tail -1 | cut -d'=' -f2)
            if [ -n "$SAVE_DIR" ]; then
                FINAL_LOG="${SAVE_DIR}/plot_features_comparison_${JOB_ID}.log"
                mv $TEMP_LOG $FINAL_LOG
                echo "  Log saved to: $FINAL_LOG" >> $MAIN_LOG
            else
                rm -f $TEMP_LOG
            fi
        else
            echo "✗ FAILED: $dataset + $model (exit code: $EXIT_CODE)" >> $MAIN_LOG
            FAIL_COUNT=$((FAIL_COUNT + 1))
            
            # 保存失败日志到logs目录
            FAIL_LOG="logs/failed_${dataset}_${model}_features_${JOB_ID}.log"
            mv $TEMP_LOG $FAIL_LOG
            echo "  Error log saved to: $FAIL_LOG" >> $MAIN_LOG
        fi
        
        echo "Progress: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed" >> $MAIN_LOG
        echo "Finished at: $(date)" >> $MAIN_LOG
        echo "" >> $MAIN_LOG
    done
done

echo "======================================" >> $MAIN_LOG
echo "All tasks completed at: $(date)" >> $MAIN_LOG
echo "Final Summary:" >> $MAIN_LOG
echo "  Total: $TOTAL_TASKS tasks" >> $MAIN_LOG
echo "  Succeeded: $SUCCESS_COUNT" >> $MAIN_LOG
echo "  Failed: $FAIL_COUNT" >> $MAIN_LOG
echo "======================================" >> $MAIN_LOG

echo "Feature comparison plot job completed. Check log: $MAIN_LOG"
