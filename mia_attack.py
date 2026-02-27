import argparse
import json
import numpy as np
import pickle
import random
import os 
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from models.base_model import BaseModel
from datasets import get_dataset
from utils.attackers import MiaAttack
from utils.utils import print_set

# --- 命令行参数定义 ---
parser = argparse.ArgumentParser(description='执行成员推断攻击 (Membership Inference Attacks)')
parser.add_argument('device', default=0, type=int, help="要使用的GPU ID")
parser.add_argument('config_path', default=0, type=str, help="配置文件路径")
parser.add_argument('--dataset_name', default='cifar10', type=str, help="数据集名称")
parser.add_argument('--model_name', default='cifar10', type=str, help="模型架构名称")
parser.add_argument('--num_cls', default=10, type=int, help="分类任务的类别数")
parser.add_argument('--input_dim', default=3, type=int, help="输入图像的通道数")
parser.add_argument('--seed', default=42, type=int, help="随机种子，用于保证实验可复现")
parser.add_argument('--epochs', default=100, type=int, help="（此脚本中未使用）受害者/影子模型的训练轮数")
parser.add_argument('--attack_epochs', default=150, type=int, help="攻击模型的训练轮数")
parser.add_argument('--batch_size', default=128, type=int, help="数据加载时的批处理大小")
parser.add_argument('--model_num', default=4, type=int, help="用于攻击的参考模型数量")
parser.add_argument('--query_num', default=8, type=int, help="对每个参考模型的查询次数，用于计算平均置信度")
parser.add_argument('--feature_group', default='G0', type=str, choices=['G0', 'G1', 'G2', 'G3', 'G4'], 
                    help="特征组合选择: G0(baseline), G1(+ref stats), G2(+z_score), G3(+ratio), G4(+interactions)")
parser.add_argument('--attack_model', default='mia_fc', type=str, 
                    help="攻击模型架构 (默认: mia_fc)")
def main(args):
    # --- 初始化环境 ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    # 设置CUDNN以保证确定性，这对于可复现性很重要
    cudnn.deterministic = True
    cudnn.benchmark = False

    # 定义结果保存和加载的基础路径
    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    print(f"基础文件夹: {base_folder}")

    # --- 加载数据集和数据索引 ---
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    aug_trainset = get_dataset(args.dataset_name, train=True, augment=True) # 带数据增强的训练集
    aug_testset = get_dataset(args.dataset_name, train=False, augment=True) # 带数据增强的测试集

    # 合并训练集和测试集以创建总数据集
    if testset is None:
        total_dataset = trainset
        aug_total_dataset = aug_trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
        aug_total_dataset = ConcatDataset([aug_trainset, aug_testset])

    total_size = len(total_dataset)
    # 从预存的文件中加载数据分区索引
    data_path = f"{base_folder}/data_index.pkl"
    with open(data_path, 'rb') as f:
        victim_train_list, victim_test_list, attack_train_list, attack_test_list, \
            tuning_train_list, tuning_test_list = pickle.load(f)
    print(f"总数据量: {total_size}")

    # --- 加载并评估受害者模型 (Victim Model) ---
    # 受害者模型是我们的攻击目标
    aug_victim_train_dataset = Subset(aug_total_dataset, victim_train_list) # 受害者模型的训练集（成员）
    aug_victim_test_dataset = Subset(aug_total_dataset, victim_test_list)   # 受害者模型的测试集（非成员）
    print(f"受害者模型训练集大小: {len(victim_train_list)}, "
          f"受害者模型测试集大小: {len(victim_test_list)}")

    # 为受害者模型的成员和非成员数据创建加载器
    aug_victim_train_loader = DataLoader(aug_victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    aug_victim_test_loader = DataLoader(aug_victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # 加载预训练好的受害者模型
    victim_model_path = f"{base_folder}/victim_model/best.pth"
    print(f"从 {victim_model_path} 加载受害者模型")
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path, True)
    # 在其自己的训练集和测试集上评估性能，以验证模型是否加载成功
    victim_model.test(aug_victim_train_loader, "受害者模型在其训练集上")
    victim_model.test(aug_victim_test_loader, "受害者模型在其测试集上")

    # --- 加载并评估影子模型 (Shadow Model) ---
    # 影子模型用于训练攻击模型
    aug_shadow_train_dataset = Subset(aug_total_dataset, attack_train_list) # 影子模型的训练集（成员）
    aug_shadow_test_dataset = Subset(aug_total_dataset, attack_test_list)   # 影子模型的测试集（非成员）
    print(f"影子模型训练集大小: {len(attack_train_list)}, "
          f"影子模型测试集大小: {len(attack_test_list)}")

    # 为影子模型的成员和非成员数据创建加载器
    aug_shadow_train_loader = DataLoader(aug_shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    aug_shadow_test_loader = DataLoader(aug_shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # 加载预训练好的影子模型
    shadow_model_path = f"{base_folder}/shadow_model/best.pth"
    print(f"从 {shadow_model_path} 加载影子模型")
    shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    shadow_model.load(shadow_model_path, True)
    # 评估影子模型性能
    shadow_model.test(aug_shadow_train_loader, "影子模型在其训练集上")
    shadow_model.test(aug_shadow_test_loader, "影子模型在其测试集上")

    print(f"用于RAPID参考模型的数据池大小 (训练): {len(tuning_train_list)}")
    print(f"用于RAPID参考模型的数据池大小 (测试): {len(tuning_test_list)}")

    # --- 使用参考模型群为RAPID攻击准备特征 ---
    # 这些列表将存储从参考模型群中提取的、用于增强攻击的特征
    rapid_victim_in_model_list = []  # 受害者模型成员数据，在参考模型群上的输出
    rapid_shadow_in_model_list = []  # 影子模型成员数据，在参考模型群上的输出
    rapid_victim_out_model_list = [] # 受害者模型非成员数据，在参考模型群上的输出
    rapid_shadow_out_model_list = [] # 影子模型非成员数据，在参考模型群上的输出

    # 遍历每一个参考模型
    for idx in range(args.model_num):
        # 每个参考模型对四组数据的置信度列表
        rapid_victim_in_confidence_list = []
        rapid_victim_out_confidence_list = []
        rapid_shadow_in_confidence_list = []
        rapid_shadow_out_confidence_list = []
        
        # 注意：在原始RAPID论文中，受害者和影子的参考模型是分开训练的。
        # 在此实现中，它们共享同一组参考模型，因此加载路径相同。
        victim_save_folder = f"{base_folder}/RAPID/reference_model_{idx}"
        shadow_save_folder = f"{base_folder}/RAPID/reference_model_{idx}"

        # 检查参考模型文件是否存在
        rapid_victim_model_path = f"{victim_save_folder}/best.pth"
        if not os.path.exists(rapid_victim_model_path):
            print(f"错误: 参考模型文件不存在: {rapid_victim_model_path}")
            print(f"请先训练 {args.model_num} 个参考模型，或减少 --model_num 参数")
            exit(1)
        
        # 加载第 idx 个参考模型 (作为受害者的参考)
        print(f"加载RAPID受害者参考模型 [{idx+1}/{args.model_num}]: {rapid_victim_model_path}")
        rapid_victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        rapid_victim_model.load(rapid_victim_model_path, True)

        # 加载第 idx 个参考模型 (作为影子的参考)
        rapid_shadow_model_path = f"{shadow_save_folder}/best.pth"
        print(f"加载RAPID影子参考模型: {rapid_shadow_model_path}")
        rapid_shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        rapid_shadow_model.load(rapid_shadow_model_path)
        
        # 对每个参考模型进行多次查询，以平滑输出（例如，如果模型中存在随机性如Dropout）
        for query in range(args.query_num):
            
            # 1. 计算当前参考模型在“受害者模型”的成员/非成员数据上的损失
            rapid_victim_in_confidences, _ = rapid_victim_model.predict_target_loss(aug_victim_train_loader)
            rapid_victim_out_confidences, _ = rapid_victim_model.predict_target_loss(aug_victim_test_loader)
            rapid_victim_in_confidence_list.append(rapid_victim_in_confidences)
            rapid_victim_out_confidence_list.append(rapid_victim_out_confidences)

            # 2. 计算当前参考模型在“影子模型”的成员/非成员数据上的损失
            rapid_attack_in_confidences, _ = rapid_shadow_model.predict_target_loss(aug_shadow_train_loader)
            rapid_attack_out_confidences, _ = rapid_shadow_model.predict_target_loss(aug_shadow_test_loader)
            rapid_shadow_in_confidence_list.append(rapid_attack_in_confidences)
            rapid_shadow_out_confidence_list.append(rapid_attack_out_confidences)
                
        # --- 特征聚合 ---
        # 对 query_num 次查询的结果取平均，得到该模型对各数据集的稳定输出
        # 然后将这个聚合后的特征添加到对应列表中
        rapid_victim_in_model_list.append(torch.cat(rapid_victim_in_confidence_list, dim=1).mean(dim=1, keepdim=True))
        rapid_victim_out_model_list.append(torch.cat(rapid_victim_out_confidence_list, dim=1).mean(dim=1, keepdim=True))
        rapid_shadow_in_model_list.append(torch.cat(rapid_shadow_in_confidence_list, dim=1).mean(dim=1, keepdim=True))
        rapid_shadow_out_model_list.append(torch.cat(rapid_shadow_out_confidence_list, dim=1).mean(dim=1, keepdim=True))
        
        # 显式释放GPU内存 (避免加载大量模型时显存泄漏)
        del rapid_victim_model, rapid_shadow_model
        torch.cuda.empty_cache()
        
        # 定期打印内存使用情况 (每8个模型)
        if (idx + 1) % 8 == 0:
            print(f"已加载 {idx+1}/{args.model_num} 个参考模型, GPU内存: {torch.cuda.memory_allocated()/1024**2:.0f}MB / {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

    # --- 初始化并执行攻击 ---
    attacker = MiaAttack(
        victim_model, aug_victim_train_loader, aug_victim_test_loader,
        shadow_model, aug_shadow_train_loader, aug_shadow_test_loader,
        # 将所有参考模型的输出进行最终聚合（取平均），作为RAPID攻击的额外特征
        # 注意: 这里传入原始的参考模型列表(未聚合),以便计算统计特征
        verify_victim_model_list=[rapid_victim_in_model_list, rapid_victim_out_model_list],
        verify_shadow_model_list=[rapid_shadow_in_model_list, rapid_shadow_out_model_list],
        device=device, num_cls=args.num_cls, epochs=args.attack_epochs, batch_size=args.batch_size, 
        lr=0.0002, weight_decay=5e-4, optimizer="adam", scheduler="", 
        dataset_name=args.dataset_name, model_name=args.model_name, query_num=args.query_num,
        feature_group=args.feature_group, attack_model_type=args.attack_model)

    print("开始执行成员推断攻击 (RAPID)...")
    # 调用RAPID攻击方法
    pr_loss_tpr, pr_loss_auc, pr_loss_acc = attacker.rapid_attack()
    # 打印攻击结果：包括在特定FPR下的TPR、AUC和准确率
    print(f"RAPID 攻击结果 ({args.attack_model}): tpr@0.1%fpr = {pr_loss_tpr*100:.2f}, auc = {pr_loss_auc:.3f}, accuracy = {pr_loss_acc:.3f}%")


# ... main 函数 ...

if __name__ == '__main__':
    # 解析命令行参数
    args = parser.parse_args()
    
    # 从JSON配置文件中加载参数
    with open(args.config_path) as f:
        config_dict = json.load(f)
    
    # 只更新那些在命令行中没有显式指定的参数
    # 保存命令行显式指定的参数
    import sys
    cmdline_args = set()
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            param_name = arg[2:]
            cmdline_args.add(param_name)
    
    # 用JSON配置更新args，但保留命令行显式指定的参数
    for key, value in config_dict.items():
        if key not in cmdline_args and hasattr(args, key):
            setattr(args, key, value)

    # 打印最终使用的所有参数
    print_set(args)
    # 执行主函数
    main(args)