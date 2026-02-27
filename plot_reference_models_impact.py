"""
生成参考模型数量对攻击性能影响的图表
根据消融实验结果,绘制不同特征组(G0-G4)在不同参考模型数量下的TPR@0.1%FPR
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import re

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

def parse_args():
    parser = argparse.ArgumentParser(description='Plot TPR vs Number of Reference Models')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='数据集名称 (例如: cifar10, cifar100, svhn, cinic)')
    parser.add_argument('--model', type=str, required=True,
                        help='模型架构 (例如: vgg16, resnet50, densenet121, mobilenetv2)')
    parser.add_argument('--fpr_threshold', type=float, default=0.001,
                        help='FPR阈值 (默认: 0.001 = 0.1%%)')
    parser.add_argument('--groups', type=str, default='G0,G1,G2,G3,G4',
                        help='要绘制的特征组,逗号分隔 (默认: G0,G1,G2,G3,G4)')
    parser.add_argument('--model_numbers', type=str, default='1,2,4,8,16,32,64',
                        help='参考模型数量列表,逗号分隔 (默认: 1,2,4,8,16,32,64)')
    parser.add_argument('--query_num', type=int, default=None,
                        help='查询次数 (默认: None=自动选择最大query_num)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: results/{dataset}_{model}/RAPID/plots)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='输出图像DPI (默认: 300)')
    parser.add_argument('--figsize', type=str, default='8,6',
                        help='图像尺寸,逗号分隔 (默认: 8,6)')
    
    return parser.parse_args()


def calculate_tpr_at_fpr(roc_label, roc_score, target_fpr=0.001):
    """
    计算指定FPR下的TPR值
    
    使用与plot_tpr_vs_architecture.py相同的算法:
    选择FPR < target_fpr的最后一个点
    
    Args:
        roc_label: 真实标签 (1=member, 0=non-member)
        roc_score: 预测分数
        target_fpr: 目标FPR阈值
    
    Returns:
        TPR值,如果无法计算则返回None
    """
    try:
        # 处理NaN值
        roc_score_clean = np.nan_to_num(roc_score, nan=np.nanmean(roc_score))
        fpr, tpr, thresholds = roc_curve(roc_label, roc_score_clean, pos_label=1)
        
        # 使用与plot_tpr_vs_architecture.py相同的算法
        idx = np.where(fpr < target_fpr)[0]
        if len(idx):
            return tpr[idx[-1]]
        else:
            return 0.0
    except Exception as e:
        print(f"计算TPR时出错: {e}")
        return None


def find_attack_result(base_folder, feature_group, model_num, query_num=None):
    """
    查找指定参数的攻击结果文件
    
    数据来源说明:
    1. 优先查找消融实验结果: results/{dataset}_{model}/RAPID/rapid_attack_{group}_r{model_num}_q{query_num}/
    2. 如果未找到,尝试查找基线结果: results/{dataset}_{model}/RAPID/rapid_attack_{group}/
    3. 当query_num=None时,自动选择最大的query_num
    4. 数据文件: Roc_confidence_score.npz (包含ROC_label和ROC_confidence_score)
    
    Args:
        base_folder: 结果基础路径
        feature_group: 特征组 (G0-G4)
        model_num: 参考模型数量
        query_num: 查询次数 (None=自动选择最大query_num)
    
    Returns:
        tuple: (folder_path, actual_query_num) 或 (None, None)
    """
    rapid_folder = f"{base_folder}/RAPID"
    
    if not os.path.exists(rapid_folder):
        return None, None
    
    # 构建正则表达式匹配模式
    if query_num is None:
        # 自动匹配: rapid_attack_G0_r16_q*
        pattern = f"rapid_attack_{feature_group}_r{model_num}_q\\d+"
    else:
        # 精确匹配指定的query_num
        pattern = f"rapid_attack_{feature_group}_r{model_num}_q{query_num}$"
    
    # 遍历RAPID文件夹,找到匹配的文件夹
    best_folder = None
    max_query_num = -1
    
    for folder_name in os.listdir(rapid_folder):
        if re.match(pattern, folder_name):
            folder_path = os.path.join(rapid_folder, folder_name)
            result_file = os.path.join(folder_path, "Roc_confidence_score.npz")
            
            if os.path.exists(result_file):
                # 提取query_num
                match = re.search(r'_q(\d+)', folder_name)
                if match:
                    found_query_num = int(match.group(1))
                    
                    if query_num is None:
                        # 选择query_num最大的
                        if found_query_num > max_query_num:
                            max_query_num = found_query_num
                            best_folder = folder_path
                    else:
                        # 精确匹配
                        if found_query_num == query_num:
                            return folder_path, found_query_num
    
    if best_folder is not None:
        return best_folder, max_query_num
    
    # 不使用基线结果作为消融实验的替代
    # 如果指定了model_num,必须找到对应的消融实验结果
    return None, None


def collect_data(dataset, model, groups, model_numbers, fpr_threshold, query_num=None):
    """
    收集所有实验数据
    
    Args:
        dataset: 数据集名称
        model: 模型名称
        groups: 特征组列表
        model_numbers: 参考模型数量列表
        fpr_threshold: FPR阈值
        query_num: 查询次数 (None=自动选择)
    
    Returns:
        tuple: (data, actual_query_nums)
            data: {group: {model_num: tpr_value}}
            actual_query_nums: {group: {model_num: query_num}}
    """
    base_folder = f"results/{dataset}_{model}"
    
    data = {}
    actual_query_nums = {}
    
    for group in groups:
        data[group] = {}
        actual_query_nums[group] = {}
        
        for model_num in model_numbers:
            folder, found_query_num = find_attack_result(base_folder, group, model_num, query_num)
            
            if folder is None:
                print(f"警告: 未找到 {group}, model_num={model_num}" + 
                      (f", query_num={query_num}" if query_num else "") + " 的结果")
                continue
            
            # 显示数据来源
            folder_name = os.path.basename(folder)
            print(f"数据来源: {folder}")
            
            result_file = os.path.join(folder, "Roc_confidence_score.npz")
            
            try:
                result_data = np.load(result_file)
                roc_label = result_data['ROC_label']
                roc_score = result_data['ROC_confidence_score']
                
                tpr = calculate_tpr_at_fpr(roc_label, roc_score, fpr_threshold)
                
                if tpr is not None:
                    data[group][model_num] = tpr
                    actual_query_nums[group][model_num] = found_query_num
                    
                    query_info = f", query_num={found_query_num}" if found_query_num else ""
                    print(f"✓ {group}, model_num={model_num}{query_info}: TPR@{fpr_threshold*100:.1f}%FPR = {tpr*100:.2f}%")
                else:
                    print(f"✗ {group}, model_num={model_num}: 无法计算TPR")
                    
            except Exception as e:
                print(f"✗ {group}, model_num={model_num}: 加载失败 - {e}")
    
    return data, actual_query_nums


def plot_results(data, actual_query_nums, model_numbers, fpr_threshold, args):
    """
    绘制结果图表
    
    Args:
        data: TPR数据
        actual_query_nums: 实际使用的查询次数
        model_numbers: 参考模型数量列表
        fpr_threshold: FPR阈值
        args: 命令行参数
    """
    figsize = tuple(map(float, args.figsize.split(',')))
    plt.figure(figsize=figsize)
    
    # 颜色映射
    colors = {
        'G0': '#1f77b4',  # 蓝色
        'G1': '#ff7f0e',  # 橙色
        'G2': '#2ca02c',  # 绿色
        'G3': '#d62728',  # 红色
        'G4': '#9467bd',  # 紫色
    }
    
    # 标记样式
    markers = {
        'G0': 'o',
        'G1': 's',
        'G2': '^',
        'G3': 'D',
        'G4': 'v',
    }
    
    # 绘制每条曲线
    for group in sorted(data.keys()):
        group_data = data[group]
        
        # 获取该组的所有数据点
        x_vals = []
        y_vals = []
        
        for model_num in sorted(group_data.keys()):
            x_vals.append(model_num)
            y_vals.append(group_data[model_num] * 100)  # 转换为百分比
        
        if x_vals:
            plt.plot(x_vals, y_vals, 
                    marker=markers.get(group, 'o'),
                    linestyle='--',
                    linewidth=2,
                    markersize=8,
                    color=colors.get(group),
                    label=f'RAPID_{group}',
                    alpha=0.8)
    
    # 设置坐标轴
    plt.xlabel('Number of reference models', fontsize=14)
    plt.ylabel(f'TPR @{fpr_threshold*100:.1f}% FPR (%)', fontsize=14)
    
    # 设置x轴为对数刻度
    plt.xscale('log')
    plt.xticks(model_numbers, [str(n) for n in model_numbers])
    
    # 网格和图例
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=11)
    
    # 检查所有query_num是否一致（用于文件名）
    all_query_nums = set()
    for group in actual_query_nums.values():
        for qnum in group.values():
            if qnum is not None:
                all_query_nums.add(qnum)
    
    plt.tight_layout()
    
    # 保存图片
    if args.output_dir is None:
        output_dir = f"results/{args.dataset}_{args.model}/RAPID/plots"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    fpr_str = f"{fpr_threshold*100:.1g}".replace('.', '_')
    
    # 文件名中包含query_num信息
    if args.query_num is not None:
        filename = f'tpr_vs_reference_models_{args.dataset}_{args.model}_fpr{fpr_str}_q{args.query_num}.pdf'
    elif len(all_query_nums) == 1:
        filename = f'tpr_vs_reference_models_{args.dataset}_{args.model}_fpr{fpr_str}_q{list(all_query_nums)[0]}.pdf'
    else:
        filename = f'tpr_vs_reference_models_{args.dataset}_{args.model}_fpr{fpr_str}.pdf'
    
    output_file = os.path.join(output_dir, filename)
    
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print(f"\n图表已保存至: {output_file}")
    
    # 同时保存PNG格式
    output_file_png = output_file.replace('.pdf', '.png')
    plt.savefig(output_file_png, dpi=args.dpi, bbox_inches='tight')
    print(f"图表已保存至: {output_file_png}")
def main():
    args = parse_args()
    
    # 解析参数
    groups = [g.strip() for g in args.groups.split(',')]
    model_numbers = [int(n.strip()) for n in args.model_numbers.split(',')]
    
    print(f"=" * 60)
    print(f"参考模型数量影响分析")
    print(f"=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"FPR阈值: {args.fpr_threshold*100:.1f}%")
    print(f"特征组: {groups}")
    print(f"参考模型数量: {model_numbers}")
    print(f"查询次数: {args.query_num if args.query_num else '自动选择(最大)'}")
    print(f"\n数据来源说明:")
    print(f"1. 搜索路径: results/{args.dataset}_{args.model}/RAPID/")
    print(f"2. 消融实验文件夹格式: rapid_attack_{{group}}_r{{model_num}}_q{{query_num}}/")
    print(f"3. 基线文件夹格式: rapid_attack_{{group}}/")
    print(f"4. 数据文件: Roc_confidence_score.npz")
    print(f"=" * 60)
    
    # 收集数据
    print(f"\n正在收集实验数据...")
    data, actual_query_nums = collect_data(args.dataset, args.model, groups, model_numbers, 
                                           args.fpr_threshold, args.query_num)
    
    # 检查是否有数据
    if not any(data.values()):
        print("\n错误: 未找到任何实验数据!")
        return
    
    # 绘制结果
    print("\n正在绘制图表...")
    plot_results(data, actual_query_nums, model_numbers, args.fpr_threshold, args)


if __name__ == '__main__':
    main()
