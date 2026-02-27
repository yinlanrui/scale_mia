# CIFAR10 VGG16
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
python refer_model.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 4
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json --model_num 4 --query_num 8
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack

# ========== 跨架构攻击示例 ==========
# 1. 同架构攻击（应该与上面的mia_attack.py结果相同）
python mia_attack_cross_arch.py 0 config/cifar10/cifar10_vgg16.json \
    --victim_model_name vgg16 \
    --shadow_model_name vgg16 \
    --reference_model_name vgg16 \
    --feature_group G4

# 2. 跨架构攻击（VGG16作为影子和参考，攻击ResNet50）
python mia_attack_cross_arch.py 0 config/cifar10/cifar10_resnet50.json \
    --victim_model_name resnet50 \
    --shadow_model_name vgg16 \
    --reference_model_name vgg16 \
    --feature_group G4

# 3. 跨架构攻击（ResNet50作为影子和参考，攻击VGG16）
python mia_attack_cross_arch.py 0 config/cifar10/cifar10_vgg16.json \
    --victim_model_name vgg16 \
    --shadow_model_name resnet50 \
    --reference_model_name resnet50 \
    --feature_group G4



# 使用 --feature_group all（最简洁）
python plot.py 0 config/cifar10/cifar10_mobilenetv2.json --dataset_name cifar10 --model_name mobilenetv2 --feature_group all
# 明确指定所有攻击
python plot.py 0 config/cifar10/cifar10_mobilenetv2.json --dataset_name cifar10 --model_name mobilenetv2 --attacks rapid_attack_G0,rapid_attack_G1,rapid_attack_G2,rapid_attack_G3,rapid_attack_G4
# 方式3：只绘制部分特征组合

python plot.py 0 config/svhn/svhn_resnet50.json --dataset_name svhn --model_name resnet50 --attacks rapid_attack_G0,rapid_attack_G4 

# 只对比 G0, G2, G4
python plot.py 0 config/cifar10/cifar10_resnet50.json --dataset_name cifar10 --model_name resnet50 --attacks rapid_attack_G0,rapid_attack_G2,rapid_attack_G4


# 1. 单个攻击 (新修复)
python plot.py 0 config/cifar10/cifar10_vgg16.json \
    --attacks rapid_attack_G0_mia_fc

# 2. 多个攻击对比
python plot.py 0 config/cifar10/cifar10_vgg16.json \
    --attacks rapid_attack_G0_mia_fc,rapid_attack_G0_mia_fc_bn,rapid_attack_G0_mia_enhanced,rapid_attack_G0_mia_attention

# 3. 使用feature_group参数 (单个)
python plot.py 0 config/cifar10/cifar10_vgg16.json \
    --feature_group G0

# 4. 自动绘制所有特征组
python plot.py 0 config/cifar10/cifar10_vgg16.json \
    --feature_group all

# 5. 多数据集对比
python plot.py 0 config/cifar10/cifar10_vgg16.json \
    --plot_mode datasets \
    --datasets cifar10,cifar100,cinic,svhn \
    --attacks rapid_attack_G0_mia_fc

# 6. 传统用法 (不带攻击模型后缀)
python plot.py 0 config/cifar10/cifar10_vgg16.json \
    --attacks rapid_attack_G0,rapid_attack_G1,rapid_attack_G2


# 1. 默认模式(只分析核心攻击类型: rapid_attack, G0-G4)
# 分析所有数据集-模型的核心攻击类型
python generate_attack_logs.py

# 或者分析单个数据集-模型的核心攻击类型
python generate_attack_logs.py --dataset_model cifar10_densenet121

# 2. 分析所有攻击类型(包括实验性类型如G8, G10, L1, L2, combo等)

# 分析所有数据集-模型的所有攻击类型
python generate_attack_logs.py --all_attacks

# 或者分析单个数据集-模型的所有攻击类型
python generate_attack_logs.py --dataset_model cifar10_densenet121 --all_attacks

# 3. 分析单个攻击类型
python generate_attack_logs.py --dataset_model cifar10_densenet121 --attack_type rapid_attack_G0


# 分析单个数据集的消融实验
python generate_attack_logs.py --dataset_model cifar10_vgg16 --ablation

# 分析所有数据集的消融实验
python generate_attack_logs.py --ablation

# 1. 改数据集为CIFAR-100
python plot_tpr_vs_architecture.py --dataset cifar100

# 2. 改FPR为0.01%
python plot_tpr_vs_architecture.py --fpr 0.001

# 3. 对比G0、G2、G4三组
python plot_tpr_vs_architecture.py --groups "0,2,4"

# 4. 只看VGG16和ResNet50
python plot_tpr_vs_architecture.py --archs "vgg16,resnet50"

# 5. 组合多个参数
python plot_tpr_vs_architecture.py --dataset svhn --fpr 0.0001 --groups "0,3,4" --archs "vgg16,densenet121"


# 基本用法 - 自动选择最大query_num
python plot_reference_models_impact.py --dataset cifar10 --model vgg16

# 指定查询次数
python plot_reference_models_impact.py --dataset cifar10 --model vgg16 --query_num 64

# 只绘制G0和G4
python plot_reference_models_impact.py --dataset cifar10 --model vgg16 --groups "G0,G4"

# 自定义参考模型数量
python plot_reference_models_impact.py --dataset cifar10 --model vgg16 --model_numbers "1,4,16,64"

python plot_reference_models_impact.py --dataset cifar10 --model vgg16 --model_numbers "1,2,4,8,16,32,64" --query_num 64


# 基础用法：分析 CIFAR10-VGG16，固定参考模型数=16
python plot_query_nums_impact.py --dataset cifar10 --model vgg16 --ref_model 16

# 自定义查询次数范围
python plot_query_nums_impact.py --dataset cifar10 --model vgg16 --ref_model 16 \
    --query_nums "1,4,16,64"

# 只对比 G0 和 G4
python plot_query_nums_impact.py --dataset cifar10 --model vgg16 --ref_model 16 \
    --groups "G0,G4"

# 修改 FPR 阈值
python plot_query_nums_impact.py --dataset cifar10 --model vgg16 --ref_model 16 \
    --fpr_threshold 0.0001  # 0.01% FPR



# 1. 绘制单个配置的ROC曲线
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G4" --ref_models "1" --query_nums "1"

python plot_ablation_roc.py 0 config/svhn/svhn_resnet50.json --groups "G0,G4" --ref_models "4" --query_nums "8"

python plot_ablation_roc.py 0 config/flowers102/flowers102_resnet50.json --groups "G0,G1,G2,G3,G4" --ref_models "4" --query_nums "8" 
# 2. 对比不同策略 (G0 vs G4)
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G0,G4" --ref_models "16" --query_nums "64"

# 3. 对比不同参考模型数量
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G4" --ref_models "1,8,16,32,64" --query_nums "64"

# 4. 对比不同查询次数
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G4" --ref_models "16" --query_nums "1,8,16,32,64"

# 5. 自动查找最大query_num
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G0,G1,G2,G3,G4" --ref_models "16" --auto_find

# 6. 完整对比 (所有策略 × 多个参考模型数量)
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G0,G1,G2,G3,G4" --ref_models "1,16,64" --query_nums "64"

# 7. 自定义坐标轴范围
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json --groups "G4" --ref_models "16" --query_nums "64" --xlim "1e-4,1e-2" --ylim "1e-3,1e-1"

# 只检查指定的组合
python check_model_performance.py --combinations cifar10_vgg16,cifar10_resnet50


#画直方图
python plot_score_distribution.py


# 使用默认参数 (r=4, q=8)
python compare_groups.py --dataset_model cifar10_vgg16

# 自定义参数
python compare_groups.py --dataset_model cifar10_vgg16 --ref_model 16 --query_num 64

# 批量分析所有数据集，使用自定义参数
python compare_groups.py --ref_model 8 --query_num 16



# 生成所有图表 (默认CIFAR-10, G4特征组)
python plot_cross_arch.py

# 指定数据集和特征组
python plot_cross_arch.py --dataset cifar10 --group G4

# 只生成散点图
python plot_cross_arch.py --plot_type scatter

# 指定输出目录
python plot_cross_arch.py --output_dir my_plots

# 生成CIFAR-100的图表
python plot_cross_arch.py --dataset cifar100 --group G3



python plot_baseline_roc.py


python evaluate_baseline_metrics.py

python convert_image_to_pdf.py <图片路径>
