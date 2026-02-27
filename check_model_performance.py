"""
检查所有数据集-模型组合的受害者模型和影子模型在训练集和测试集上的性能
用于分析模型过拟合程度,帮助理解 G1-G4 策略的适用条件
"""

import argparse
import os
import pickle
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from models.base_model import BaseModel
from datasets import get_dataset
from tabulate import tabulate

def evaluate_model_directly(model, data_loader, device='cuda:0'):
    """直接评估模型性能,不依赖 BaseModel.test() 的打印"""
    model.model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    acc = 100. * correct / total
    total_loss /= total
    
    return acc, total_loss

def evaluate_model(model, train_loader, test_loader, model_name, device='cuda:0'):
    """评估模型在训练集和测试集上的性能"""
    # 评估训练集
    train_acc, train_loss = evaluate_model_directly(model, train_loader, device)
    # 评估测试集
    test_acc, test_loss = evaluate_model_directly(model, test_loader, device)
    
    return {
        'model_name': model_name,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'overfit_gap': train_acc - test_acc
    }

def check_dataset_model(dataset_name, model_name, device='cuda:0', verbose=False):
    """检查单个数据集-模型组合的性能"""
    base_folder = f"results/{dataset_name}_{model_name}"
    
    # 检查文件夹是否存在
    if not os.path.exists(base_folder):
        return None
    
    # 检查配置文件
    config_path = f"config/{dataset_name}/{dataset_name}_{model_name}.json"
    if not os.path.exists(config_path):
        if verbose:
            print(f"  配置文件不存在: {config_path}")
        return None
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_cls = config.get('num_cls', 10)
    input_dim = config.get('input_dim', 3)
    batch_size = config.get('batch_size', 128)
    
    try:
        # 加载数据集
        trainset = get_dataset(dataset_name, train=True)
        testset = get_dataset(dataset_name, train=False)
        aug_trainset = get_dataset(dataset_name, train=True, augment=True)
        aug_testset = get_dataset(dataset_name, train=False, augment=True)
        
        if testset is None:
            aug_total_dataset = aug_trainset
        else:
            aug_total_dataset = ConcatDataset([aug_trainset, aug_testset])
        
        # 加载数据分区索引
        data_index_path = f"{base_folder}/data_index.pkl"
        if not os.path.exists(data_index_path):
            if verbose:
                print(f"  数据索引文件不存在: {data_index_path}")
            return None
        
        with open(data_index_path, 'rb') as f:
            victim_train_list, victim_test_list, attack_train_list, attack_test_list, \
                tuning_train_list, tuning_test_list = pickle.load(f)
        
        results = {}
        
        # --- 评估受害者模型 ---
        victim_model_path = f"{base_folder}/victim_model/best.pth"
        if os.path.exists(victim_model_path):
            # 创建数据加载器
            victim_train_dataset = Subset(aug_total_dataset, victim_train_list)
            victim_test_dataset = Subset(aug_total_dataset, victim_test_list)
            victim_train_loader = DataLoader(victim_train_dataset, batch_size=batch_size, 
                                            shuffle=False, num_workers=4, pin_memory=False)
            victim_test_loader = DataLoader(victim_test_dataset, batch_size=batch_size, 
                                           shuffle=False, num_workers=4, pin_memory=False)
            
            # 加载模型
            victim_model = BaseModel(model_name, num_cls=num_cls, input_dim=input_dim, device=device)
            victim_model.load(victim_model_path, verbose=False)
            
            # 评估
            results['victim'] = evaluate_model(victim_model, victim_train_loader, 
                                              victim_test_loader, 'Victim', device)
            
            del victim_model
            torch.cuda.empty_cache()
        
        # --- 评估影子模型 ---
        shadow_model_path = f"{base_folder}/shadow_model/best.pth"
        if os.path.exists(shadow_model_path):
            # 创建数据加载器
            shadow_train_dataset = Subset(aug_total_dataset, attack_train_list)
            shadow_test_dataset = Subset(aug_total_dataset, attack_test_list)
            shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=4, pin_memory=False)
            shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4, pin_memory=False)
            
            # 加载模型
            shadow_model = BaseModel(model_name, num_cls=num_cls, input_dim=input_dim, device=device)
            shadow_model.load(shadow_model_path, verbose=False)
            
            # 评估
            results['shadow'] = evaluate_model(shadow_model, shadow_train_loader,
                                              shadow_test_loader, 'Shadow', device)
            
            del shadow_model
            torch.cuda.empty_cache()
        
        if results:
            results['dataset'] = dataset_name
            results['model'] = model_name
            return results
        else:
            return None
            
    except Exception as e:
        if verbose:
            print(f"  错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='检查所有模型的训练集/测试集性能')
    parser.add_argument('--device', default='cuda:0', type=str, help='使用的设备')
    parser.add_argument('--results_dir', default='results', type=str, help='结果目录')
    parser.add_argument('--datasets', default=None, type=str, 
                       help='指定数据集,逗号分隔 (留空则检查所有)')
    parser.add_argument('--models', default=None, type=str,
                       help='指定模型,逗号分隔 (留空则检查所有)')
    parser.add_argument('--combinations', default=None, type=str,
                       help='指定数据集-模型组合,格式: dataset1_model1,dataset2_model2 (例如: cifar10_vgg16,gtsrb_resnet50)')
    parser.add_argument('--output', default=None, type=str, help='保存结果到文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 扫描 results 目录
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录不存在: {args.results_dir}")
        return
    
    # 查找所有数据集-模型组合
    dataset_models = []
    
    # 如果指定了具体组合,直接使用
    if args.combinations:
        for combo in args.combinations.split(','):
            combo = combo.strip()
            if '_' in combo:
                parts = combo.rsplit('_', 1)
                if len(parts) == 2:
                    dataset, model = parts
                    dataset_models.append((dataset, model))
                else:
                    print(f"警告: 无效的组合格式 '{combo}', 应该是 'dataset_model' 格式")
            else:
                print(f"警告: 无效的组合格式 '{combo}', 缺少下划线分隔符")
    else:
        # 扫描 results 目录
        for item in os.listdir(args.results_dir):
            item_path = os.path.join(args.results_dir, item)
            if os.path.isdir(item_path) and '_' in item:
                parts = item.rsplit('_', 1)
                if len(parts) == 2:
                    dataset, model = parts
                    dataset_models.append((dataset, model))
    
    # 过滤
    if args.datasets:
        filter_datasets = set(args.datasets.split(','))
        dataset_models = [(d, m) for d, m in dataset_models if d in filter_datasets]
    
    if args.models:
        filter_models = set(args.models.split(','))
        dataset_models = [(d, m) for d, m in dataset_models if m in filter_models]
    
    dataset_models = sorted(set(dataset_models))
    
    if not dataset_models:
        print("未找到任何数据集-模型组合")
        return
    
    print("=" * 100)
    print(f"找到 {len(dataset_models)} 个数据集-模型组合")
    print("=" * 100)
    
    # 收集所有结果
    all_results = []
    
    for dataset, model in dataset_models:
        print(f"\n检查: {dataset}_{model}")
        result = check_dataset_model(dataset, model, args.device, args.verbose)
        if result:
            all_results.append(result)
            
            # 打印简要信息
            if 'victim' in result:
                v = result['victim']
                print(f"  Victim - 训练: {v['train_acc']:.2f}%, 测试: {v['test_acc']:.2f}%, "
                      f"过拟合差距: {v['overfit_gap']:.2f}%")
            
            if 'shadow' in result:
                s = result['shadow']
                print(f"  Shadow - 训练: {s['train_acc']:.2f}%, 测试: {s['test_acc']:.2f}%, "
                      f"过拟合差距: {s['overfit_gap']:.2f}%")
    
    if not all_results:
        print("\n未找到任何有效结果")
        return
    
    # 生成汇总表格
    print("\n" + "=" * 100)
    print("汇总结果")
    print("=" * 100)
    
    table_data = []
    for result in all_results:
        dataset = result['dataset']
        model = result['model']
        
        if 'victim' in result and 'shadow' in result:
            v = result['victim']
            s = result['shadow']
            
            table_data.append([
                f"{dataset}_{model}",
                f"{v['train_acc']:.2f}%",
                f"{v['test_acc']:.2f}%",
                f"{v['overfit_gap']:.2f}%",
                f"{s['train_acc']:.2f}%",
                f"{s['test_acc']:.2f}%",
                f"{s['overfit_gap']:.2f}%"
            ])
    
    headers = [
        "数据集_模型",
        "Victim训练",
        "Victim测试",
        "Victim过拟合",
        "Shadow训练",
        "Shadow测试",
        "Shadow过拟合"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 分析过拟合程度与攻击成功的关系
    print("\n" + "=" * 100)
    print("过拟合程度分析")
    print("=" * 100)
    
    # 按过拟合差距分组
    low_overfit = []   # < 5%
    mid_overfit = []   # 5-15%
    high_overfit = []  # > 15%
    
    for result in all_results:
        if 'victim' in result:
            gap = result['victim']['overfit_gap']
            name = f"{result['dataset']}_{result['model']}"
            
            if gap < 5:
                low_overfit.append((name, gap))
            elif gap < 15:
                mid_overfit.append((name, gap))
            else:
                high_overfit.append((name, gap))
    
    print(f"\n低过拟合 (<5%): {len(low_overfit)} 个")
    for name, gap in sorted(low_overfit, key=lambda x: x[1]):
        print(f"  {name}: {gap:.2f}%")
    
    print(f"\n中等过拟合 (5-15%): {len(mid_overfit)} 个")
    for name, gap in sorted(mid_overfit, key=lambda x: x[1]):
        print(f"  {name}: {gap:.2f}%")
    
    print(f"\n高过拟合 (>15%): {len(high_overfit)} 个")
    for name, gap in sorted(high_overfit, key=lambda x: x[1]):
        print(f"  {name}: {gap:.2f}%")
    
    # 保存到文件
    if args.output:
        output_data = {
            'summary': table_data,
            'headers': headers,
            'all_results': all_results,
            'overfitting_analysis': {
                'low': low_overfit,
                'mid': mid_overfit,
                'high': high_overfit
            }
        }
        
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {args.output}")
    
    print("\n" + "=" * 100)
    print("分析建议:")
    print("=" * 100)
    print("✓ 过拟合差距 5-15% 的数据集最适合 G1-G4 策略")
    print("✗ 过拟合差距 <5% 的数据集(如Fashion-MNIST, EMNIST)训练集和测试集差异太小")
    print("? 过拟合差距 >15% 的数据集可能模型训练不充分或任务过难")
    print("=" * 100)

if __name__ == '__main__':
    main()
