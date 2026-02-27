"""
生成消融实验的ROC曲线图
支持指定策略(G0-G4)、参考模型数量、查询次数来绘制ROC曲线
"""

import argparse
import json
import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from utils.utils import roc_plot
import matplotlib

# --- 参数解析器定义 ---
parser = argparse.ArgumentParser(description='Ablation Study ROC Plot - Customizable Strategy/Models/Queries')

# 基础参数
parser.add_argument('device', default=0, type=int, help="要使用的GPU设备ID")
parser.add_argument('config_path', default=0, type=str, help="配置文件路径")
parser.add_argument('--dataset_name', default='cifar10', type=str, help="数据集名称")
parser.add_argument('--model_name', default='vgg16', type=str, help="模型架构名称")
parser.add_argument('--seed', default=42, type=int, help="全局随机种子")

# 消融实验参数
parser.add_argument('--groups', default='G0,G4', type=str, 
                    help="要绘制的特征组,逗号分隔 (例如: 'G0,G1,G2,G3,G4' 或 'G0,G4')")
parser.add_argument('--ref_models', default='16', type=str,
                    help="参考模型数量,逗号分隔 (例如: '1,8,16,32' 或 '16')")
parser.add_argument('--query_nums', default='64', type=str,
                    help="查询次数,逗号分隔 (例如: '1,16,32,64' 或 '64')")
parser.add_argument('--auto_find', action='store_true',
                    help="自动查找最大query_num (忽略--query_nums参数)")

# 图表设置
parser.add_argument('--xlim', default='1e-4,1', type=str, help="x轴范围,逗号分隔 (默认: 1e-4,1)")
parser.add_argument('--ylim', default='1e-4,1', type=str, help="y轴范围,逗号分隔 (默认: 1e-4,1)")
parser.add_argument('--output_dir', default=None, type=str, help="输出目录 (默认自动生成)")
parser.add_argument('--dpi', default=300, type=int, help="图像DPI (默认: 300)")
parser.add_argument('--figsize', default='8,6', type=str, help="图像尺寸,逗号分隔 (默认: 8,6)")


def find_ablation_result(base_folder, group, ref_model, query_num=None):
    """
    查找消融实验结果文件
    
    Args:
        base_folder: 基础文件夹路径
        group: 特征组 (G0-G4)
        ref_model: 参考模型数量
        query_num: 查询次数 (None=自动查找最大)
    
    Returns:
        tuple: (folder_path, actual_query_num) 或 (None, None)
    """
    rapid_folder = f"{base_folder}/RAPID"
    
    if not os.path.exists(rapid_folder):
        return None, None
    
    # 如果query_num为None,自动查找最大的query_num
    if query_num is None:
        import re
        pattern = re.compile(f"^rapid_attack_{group}_r{ref_model}_q(\\d+)$")
        max_q = -1
        best_folder = None
        
        for folder_name in os.listdir(rapid_folder):
            match = pattern.match(folder_name)
            if match:
                q = int(match.group(1))
                folder_path = os.path.join(rapid_folder, folder_name)
                result_file = os.path.join(folder_path, "Roc_confidence_score.npz")
                
                if os.path.exists(result_file) and q > max_q:
                    max_q = q
                    best_folder = folder_path
        
        if best_folder:
            return best_folder, max_q
        
        # 尝试查找基线版本 (没有r和q后缀)
        baseline_folder = f"{rapid_folder}/rapid_attack_{group}"
        baseline_file = os.path.join(baseline_folder, "Roc_confidence_score.npz")
        if os.path.exists(baseline_file):
            return baseline_folder, None
        
        return None, None
    
    # 精确匹配指定的query_num
    folder_name = f"rapid_attack_{group}_r{ref_model}_q{query_num}"
    folder_path = os.path.join(rapid_folder, folder_name)
    result_file = os.path.join(folder_path, "Roc_confidence_score.npz")
    
    if os.path.exists(result_file):
        return folder_path, query_num
    
    # 如果找不到精确匹配,且参数为r=4,q=8(默认配置),则回退到基线版本
    # rapid_attack_G4 实际上就是 r=4, q=8 的默认配置
    if ref_model == 4 and query_num == 8:
        baseline_folder = f"{rapid_folder}/rapid_attack_{group}"
        baseline_file = os.path.join(baseline_folder, "Roc_confidence_score.npz")
        if os.path.exists(baseline_file):
            return baseline_folder, query_num  # 返回query_num=8表示这是默认配置
    
    return None, None


def generate_label(group, ref_model, query_num, acc):
    """
    生成图例标签
    
    Args:
        group: 特征组
        ref_model: 参考模型数量
        query_num: 查询次数 (可以是None)
        acc: 准确率
    
    Returns:
        str: 图例标签
    """
    return f'RAPID_{group}'


def main(args):
    """
    主函数
    """
    # --- 环境设置 ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.deterministic = True
    cudnn.benchmark = False

    # --- 解析参数 ---
    groups = [g.strip() for g in args.groups.split(',')]
    ref_models = [int(r.strip()) for r in args.ref_models.split(',')]
    
    if args.auto_find:
        query_nums = [None]  # None表示自动查找
    else:
        query_nums = [int(q.strip()) for q in args.query_nums.split(',')]
    
    figsize = tuple(map(float, args.figsize.split(',')))
    xlim = tuple(map(float, args.xlim.split(',')))
    ylim = tuple(map(float, args.ylim.split(',')))
    
    print(f"=" * 60)
    print(f"消融实验ROC曲线绘制")
    print(f"=" * 60)
    print(f"数据集: {args.dataset_name}")
    print(f"模型: {args.model_name}")
    print(f"特征组: {groups}")
    print(f"参考模型数量: {ref_models}")
    if args.auto_find:
        print(f"查询次数: 自动查找最大值")
    else:
        print(f"查询次数: {query_nums}")
    print(f"=" * 60)
    
    # --- 创建图形 ---
    plt.figure(figsize=figsize)
    
    # --- 加载并绘制数据 ---
    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    loaded_count = 0
    
    for group in groups:
        for ref_model in ref_models:
            for query_num in query_nums:
                folder, actual_query_num = find_ablation_result(
                    base_folder, group, ref_model, query_num
                )
                
                if folder is None:
                    query_info = f"q={query_num}" if query_num else "auto"
                    print(f"⚠ 未找到: {group}, r={ref_model}, {query_info}")
                    continue
                
                result_file = os.path.join(folder, "Roc_confidence_score.npz")
                
                try:
                    data = np.load(result_file)
                    acc = data['acc']
                    roc_label = data['ROC_label']
                    roc_score = data['ROC_confidence_score']
                    
                    label = generate_label(group, ref_model, actual_query_num, acc)
                    roc_plot(roc_label, roc_score, label, plot=True)
                    
                    query_info = f"q={actual_query_num}" if actual_query_num else "baseline"
                    print(f"✓ 加载: {group}, r={ref_model}, {query_info}, acc={acc:.1f}%")
                    print(f"  路径: {folder}")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"✗ 加载失败: {group}, r={ref_model}, q={query_num}")
                    print(f"  错误: {e}")
    
    if loaded_count == 0:
        print("\n错误: 未找到任何有效的实验结果!")
        return None
    
    print(f"\n成功加载 {loaded_count} 个实验结果")
    
    # --- 图表定制 ---
    matplotlib.rcParams.update({'font.size': 16})
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.semilogx()
    plt.semilogy()
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    
    # --- 保存图表 ---
    save_dir = save_plot(args, groups, ref_models, query_nums)
    return save_dir


def save_plot(args, groups, ref_models, query_nums):
    """
    保存图表
    """
    # 生成输出目录
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = f"results/{args.dataset_name}_{args.model_name}/RAPID/ablation_plots"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    groups_str = '_'.join(groups)
    ref_models_str = '_'.join(map(str, ref_models))
    
    if args.auto_find:
        query_str = 'auto'
    else:
        query_str = '_'.join(map(str, query_nums))
    
    filename = f'roc_ablation_{args.dataset_name}_{args.model_name}_{groups_str}_r{ref_models_str}_q{query_str}.pdf'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\n图表已保存至: {save_path}")
    
    # 同时保存PNG格式
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight')
    print(f"图表已保存至: {png_path}")
    
    print(f"LOG_SAVE_DIR={save_dir}")
    
    return save_dir


if __name__ == '__main__':
    args = parser.parse_args()
    
    # 从配置文件加载参数
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    save_dir = main(args)
    
    if save_dir:
        print(f"\n=== 完成 ===")
        print(f"保存目录: {save_dir}")
