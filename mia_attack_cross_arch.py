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
parser = argparse.ArgumentParser(description='跨架构成员推断攻击 (Cross-Architecture MIA)')
parser.add_argument('device', default=0, type=int, help="要使用的GPU ID")
parser.add_argument('config_path', default=0, type=str, help="配置文件路径")
parser.add_argument('--dataset_name', default='cifar10', type=str, help="数据集名称")
parser.add_argument('--victim_model_name', default='resnet50', type=str, help="受害者模型架构")
parser.add_argument('--shadow_model_name', default='vgg16', type=str, help="影子模型架构")
parser.add_argument('--reference_model_name', default='vgg16', type=str, help="参考模型架构")
parser.add_argument('--num_cls', default=10, type=int, help="分类任务的类别数")
parser.add_argument('--input_dim', default=3, type=int, help="输入图像的通道数")
parser.add_argument('--seed', default=42, type=int, help="随机种子")
parser.add_argument('--attack_epochs', default=150, type=int, help="攻击模型的训练轮数")
parser.add_argument('--batch_size', default=128, type=int, help="批处理大小")
parser.add_argument('--model_num', default=4, type=int, help="参考模型数量")
parser.add_argument('--query_num', default=8, type=int, help="查询次数")
parser.add_argument('--feature_group', default='G4', type=str, choices=['G0', 'G1', 'G2', 'G3', 'G4'], 
                    help="特征组合选择")
parser.add_argument('--attack_model', default='mia_fc', type=str, help="攻击模型架构")

def main(args):
    # --- 初始化环境 ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.deterministic = True
    cudnn.benchmark = False

    print("=" * 80)
    print("跨架构成员推断攻击")
    print("=" * 80)
    print(f"受害者模型架构: {args.victim_model_name}")
    print(f"影子模型架构:   {args.shadow_model_name}")
    print(f"参考模型架构:   {args.reference_model_name}")
    print(f"数据集:         {args.dataset_name}")
    print(f"特征组合:       {args.feature_group}")
    print("=" * 80)

    # --- 加载数据集 ---
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    aug_trainset = get_dataset(args.dataset_name, train=True, augment=True)
    aug_testset = get_dataset(args.dataset_name, train=False, augment=True)

    if testset is None:
        total_dataset = trainset
        aug_total_dataset = aug_trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
        aug_total_dataset = ConcatDataset([aug_trainset, aug_testset])

    total_size = len(total_dataset)

    # --- 加载受害者模型的数据分区 ---
    victim_base_folder = f"results/{args.dataset_name}_{args.victim_model_name}"
    victim_data_path = f"{victim_base_folder}/data_index.pkl"
    
    if not os.path.exists(victim_data_path):
        print(f"错误: 未找到受害者模型数据索引: {victim_data_path}")
        print(f"请先训练受害者模型")
        exit(1)
    
    with open(victim_data_path, 'rb') as f:
        victim_train_list, victim_test_list, _, _, _, _ = pickle.load(f)
    
    print(f"受害者模型数据分区: 训练集={len(victim_train_list)}, 测试集={len(victim_test_list)}")

    # --- 加载影子模型的数据分区 ---
    shadow_base_folder = f"results/{args.dataset_name}_{args.shadow_model_name}"
    shadow_data_path = f"{shadow_base_folder}/data_index.pkl"
    
    if not os.path.exists(shadow_data_path):
        print(f"错误: 未找到影子模型数据索引: {shadow_data_path}")
        print(f"请先训练影子模型")
        exit(1)
    
    with open(shadow_data_path, 'rb') as f:
        _, _, attack_train_list, attack_test_list, _, _ = pickle.load(f)
    
    print(f"影子模型数据分区: 训练集={len(attack_train_list)}, 测试集={len(attack_test_list)}")

    # --- 创建数据加载器 ---
    aug_victim_train_dataset = Subset(aug_total_dataset, victim_train_list)
    aug_victim_test_dataset = Subset(aug_total_dataset, victim_test_list)
    aug_shadow_train_dataset = Subset(aug_total_dataset, attack_train_list)
    aug_shadow_test_dataset = Subset(aug_total_dataset, attack_test_list)

    aug_victim_train_loader = DataLoader(aug_victim_train_dataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
    aug_victim_test_loader = DataLoader(aug_victim_test_dataset, batch_size=args.batch_size, 
                                       shuffle=False, num_workers=4, pin_memory=False)

    aug_shadow_train_loader = DataLoader(aug_shadow_train_dataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
    aug_shadow_test_loader = DataLoader(aug_shadow_test_dataset, batch_size=args.batch_size, 
                                       shuffle=False, num_workers=4, pin_memory=False)

    # --- 加载受害者模型 ---
    victim_model_path = f"{victim_base_folder}/victim_model/best.pth"
    if not os.path.exists(victim_model_path):
        print(f"错误: 未找到受害者模型: {victim_model_path}")
        exit(1)
    
    print(f"\n加载受害者模型 ({args.victim_model_name}): {victim_model_path}")
    victim_model = BaseModel(args.victim_model_name, num_cls=args.num_cls, 
                            input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path, True)
    victim_model.test(aug_victim_train_loader, f"受害者模型({args.victim_model_name})在其训练集上")
    victim_model.test(aug_victim_test_loader, f"受害者模型({args.victim_model_name})在其测试集上")

    # --- 加载影子模型 ---
    shadow_model_path = f"{shadow_base_folder}/shadow_model/best.pth"
    if not os.path.exists(shadow_model_path):
        print(f"错误: 未找到影子模型: {shadow_model_path}")
        exit(1)
    
    print(f"\n加载影子模型 ({args.shadow_model_name}): {shadow_model_path}")
    shadow_model = BaseModel(args.shadow_model_name, num_cls=args.num_cls, 
                            input_dim=args.input_dim, device=device)
    shadow_model.load(shadow_model_path, True)
    shadow_model.test(aug_shadow_train_loader, f"影子模型({args.shadow_model_name})在其训练集上")
    shadow_model.test(aug_shadow_test_loader, f"影子模型({args.shadow_model_name})在其测试集上")

    # --- 加载参考模型 ---
    print(f"\n加载参考模型 (架构: {args.reference_model_name})...")
    reference_base_folder = f"results/{args.dataset_name}_{args.reference_model_name}"
    
    rapid_victim_in_model_list = []
    rapid_shadow_in_model_list = []
    rapid_victim_out_model_list = []
    rapid_shadow_out_model_list = []

    # 遍历每一个参考模型
    for idx in range(args.model_num):
        # 每个参考模型对四组数据的置信度列表
        rapid_victim_in_confidence_list = []
        rapid_victim_out_confidence_list = []
        rapid_shadow_in_confidence_list = []
        rapid_shadow_out_confidence_list = []
        
        # 参考模型路径 (使用reference_model_name指定的架构)
        reference_model_path = f"{reference_base_folder}/RAPID/reference_model_{idx}/best.pth"
        
        if not os.path.exists(reference_model_path):
            print(f"错误: 参考模型文件不存在: {reference_model_path}")
            print(f"请先训练{args.reference_model_name}架构的参考模型")
            exit(1)
        
        print(f"加载参考模型 [{idx+1}/{args.model_num}] ({args.reference_model_name}): {reference_model_path}")
        reference_model = BaseModel(args.reference_model_name, num_cls=args.num_cls, 
                                   input_dim=args.input_dim, device=device)
        reference_model.load(reference_model_path, True)
        
        # 对每个参考模型进行多次查询，以平滑输出（例如，如果模型中存在随机性如Dropout）
        for query in range(args.query_num):
            
            # 计算参考模型在受害者数据上的损失
            rapid_victim_in_confidences, _ = reference_model.predict_target_loss(aug_victim_train_loader)
            rapid_victim_out_confidences, _ = reference_model.predict_target_loss(aug_victim_test_loader)
            rapid_victim_in_confidence_list.append(rapid_victim_in_confidences)
            rapid_victim_out_confidence_list.append(rapid_victim_out_confidences)

            # 计算参考模型在影子数据上的损失
            rapid_attack_in_confidences, _ = reference_model.predict_target_loss(aug_shadow_train_loader)
            rapid_attack_out_confidences, _ = reference_model.predict_target_loss(aug_shadow_test_loader)
            rapid_shadow_in_confidence_list.append(rapid_attack_in_confidences)
            rapid_shadow_out_confidence_list.append(rapid_attack_out_confidences)
                
        # --- 特征聚合 ---
        # 对 query_num 次查询的结果取平均，得到该模型对各数据集的稳定输出
        # 然后将这个聚合后的特征添加到对应列表中
        rapid_victim_in_model_list.append(torch.cat(rapid_victim_in_confidence_list, dim=1).mean(dim=1, keepdim=True))
        rapid_victim_out_model_list.append(torch.cat(rapid_victim_out_confidence_list, dim=1).mean(dim=1, keepdim=True))
        rapid_shadow_in_model_list.append(torch.cat(rapid_shadow_in_confidence_list, dim=1).mean(dim=1, keepdim=True))
        rapid_shadow_out_model_list.append(torch.cat(rapid_shadow_out_confidence_list, dim=1).mean(dim=1, keepdim=True))
        
        # 显式释放GPU内存
        del reference_model
        torch.cuda.empty_cache()
        
        # 定期打印内存使用情况 (每8个模型)
        if (idx + 1) % 8 == 0:
            print(f"已加载 {idx+1}/{args.model_num} 个参考模型, GPU内存: {torch.cuda.memory_allocated()/1024**2:.0f}MB / {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

    # --- 执行跨架构攻击 ---
    print("\n" + "=" * 80)
    print("开始跨架构成员推断攻击...")
    print("=" * 80)
    
    # 保存路径包含跨架构信息
    save_suffix = f"cross_arch_victim_{args.victim_model_name}_shadow_{args.shadow_model_name}_ref_{args.reference_model_name}"
    
    attacker = MiaAttack(
        victim_model, aug_victim_train_loader, aug_victim_test_loader,
        shadow_model, aug_shadow_train_loader, aug_shadow_test_loader,
        verify_victim_model_list=[rapid_victim_in_model_list, rapid_victim_out_model_list],
        verify_shadow_model_list=[rapid_shadow_in_model_list, rapid_shadow_out_model_list],
        device=device, num_cls=args.num_cls, epochs=args.attack_epochs, 
        batch_size=args.batch_size, lr=0.0002, weight_decay=5e-4, 
        optimizer="adam", scheduler="", 
        dataset_name=args.dataset_name, 
        model_name=save_suffix,  # 使用跨架构标识作为保存名称
        query_num=args.query_num,
        feature_group=args.feature_group, 
        attack_model_type=args.attack_model,
        seed=args.seed)

    pr_loss_tpr, pr_loss_auc, pr_loss_acc = attacker.rapid_attack()
    
    print("\n" + "=" * 80)
    print("跨架构攻击结果")
    print("=" * 80)
    print(f"受害者架构: {args.victim_model_name}")
    print(f"影子架构:   {args.shadow_model_name}")
    print(f"参考架构:   {args.reference_model_name}")
    print(f"特征组合:   {args.feature_group}")
    print("-" * 80)
    print(f"TPR@0.1%FPR: {pr_loss_tpr*100:.2f}%")
    print(f"AUC:         {pr_loss_auc:.4f}")
    print(f"Accuracy:    {pr_loss_acc:.2f}%")
    print("=" * 80)


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