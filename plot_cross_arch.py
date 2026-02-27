"""
绘制跨架构攻击性能对比图
生成类似论文中的跨架构攻击可视化图表
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import re
from matplotlib.patches import Rectangle

def parse_cross_arch_name(folder_name):
    """解析跨架构文件夹名称"""
    pattern = r'^(.+)_(.+)-to-(.+)_attack$'
    match = re.match(pattern, folder_name)
    
    if match:
        dataset = match.group(1)
        victim = match.group(2)
        shadow = match.group(3)
        return (dataset, victim, shadow)
    
    return None

def load_attack_result(base_dir, dataset, victim, shadow, group='G4'):
    """加载攻击结果"""
    folder_name = f"{dataset}_{victim}-to-{shadow}_attack"
    result_path = os.path.join(base_dir, folder_name, 'RAPID', f'rapid_attack_{group}', 'Roc_confidence_score.npz')
    
    if not os.path.exists(result_path):
        return None
    
    try:
        data = np.load(result_path)
        ROC_label = data['ROC_label']
        ROC_confidence_score = data['ROC_confidence_score']
        
        # 处理NaN
        ROC_confidence_score = np.nan_to_num(ROC_confidence_score, nan=np.nanmean(ROC_confidence_score))
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
        
        # 计算TPR@0.1%FPR
        idx = np.where(fpr < 0.001)[0]
        tpr_at_001 = tpr[idx[-1]] if len(idx) > 0 else 0.0
        
        return tpr_at_001
    except Exception as e:
        print(f"加载失败 {result_path}: {e}")
        return None

def plot_cross_arch_matrix(base_dir='results', dataset='cifar10', group='G4', save_path='cross_arch_performance.png'):
    """
    绘制跨架构攻击性能矩阵图
    
    类似论文图表，显示不同受害者模型（Target）和影子模型（Shadow）组合的TPR@0.1%FPR
    """
    # 定义模型列表
    models = ['vgg16', 'resnet50', 'densenet121', 'mobilenetv2']
    model_labels = ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2']
    
    # 创建性能矩阵
    n_models = len(models)
    performance_matrix = np.zeros((n_models, n_models))
    
    print(f"加载跨架构攻击结果 (特征组: {group}, 数据集: {dataset})...")
    print("=" * 80)
    
    # 填充矩阵
    for i, victim in enumerate(models):
        for j, shadow in enumerate(models):
            tpr = load_attack_result(base_dir, dataset, victim, shadow, group)
            if tpr is not None:
                performance_matrix[i, j] = tpr
                status = "✓"
            else:
                performance_matrix[i, j] = 0.0
                status = "✗"
            
            print(f"{status} Target: {victim:<15} Shadow: {shadow:<15} TPR@0.1%%: {performance_matrix[i, j]*100:.2f}%")
    
    print("=" * 80)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 定义颜色和标记样式
    colors = {'vgg16': '#1f77b4', 'resnet50': '#ff7f0e', 
              'densenet121': '#2ca02c', 'mobilenetv2': '#d62728'}
    markers = {'vgg16': 'o', 'resnet50': 's', 
               'densenet121': '^', 'mobilenetv2': 'v'}
    
    # 绘制散点图
    for i, victim in enumerate(models):
        victim_data = []
        for j, shadow in enumerate(models):
            tpr_value = performance_matrix[i, j]
            if tpr_value > 0:
                ax.scatter(i, tpr_value, 
                          color=colors[shadow], 
                          marker=markers[shadow], 
                          s=200, 
                          alpha=0.7,
                          edgecolors='black',
                          linewidths=1.5,
                          zorder=3)
    
    # 设置y轴为对数坐标
    ax.set_yscale('log')
    ax.set_ylim([0.01, 0.1])  # 聚焦在10^-2到10^-1之间
    
    # 设置x轴
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Model Architecture', fontsize=14, fontweight='bold')
    
    # 设置y轴
    ax.set_ylabel('TPR@0.1%FPR', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    
    # 设置y轴刻度,更细致地显示10^-2到10^-1之间的差异
    ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    ax.set_yticklabels(['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%'])
    # ax.set_yticklabels(['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1'])

    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # 创建图例
    legend_elements = []
    for model, label in zip(models, model_labels):
        legend_elements.append(plt.scatter([], [], 
                                          color=colors[model], 
                                          marker=markers[model],
                                          s=150,
                                          edgecolors='black',
                                          linewidths=1.5,
                                          label=label))
    
    ax.legend(handles=legend_elements, 
             title='Shadow Model Architecture',
             title_fontsize=12,
             fontsize=11,
             loc='upper right',
             framealpha=0.95,
             edgecolor='black')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片 (PNG)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {save_path}")
    
    # 保存 PDF 版本
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF版本已保存至: {pdf_path}")
    
    plt.close()
    
    return performance_matrix

def plot_heatmap(base_dir='results', dataset='cifar10', group='G4', save_path='cross_arch_heatmap.png'):
    """
    绘制跨架构攻击热力图
    """
    # 定义模型列表
    models = ['vgg16', 'resnet50', 'densenet121', 'mobilenetv2']
    model_labels = ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2']
    
    # 创建性能矩阵
    n_models = len(models)
    performance_matrix = np.zeros((n_models, n_models))
    
    print(f"\n生成热力图 (特征组: {group}, 数据集: {dataset})...")
    
    # 填充矩阵
    for i, victim in enumerate(models):
        for j, shadow in enumerate(models):
            tpr = load_attack_result(base_dir, dataset, victim, shadow, group)
            if tpr is not None:
                performance_matrix[i, j] = tpr * 100  # 转换为百分比
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(performance_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('TPR@0.1%FPR (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.set_xlabel('Shadow Model Architecture', fontsize=13, fontweight='bold')
    ax.set_ylabel('Target Model Architecture', fontsize=13, fontweight='bold')
    
    # 在每个单元格中添加数值
    for i in range(n_models):
        for j in range(n_models):
            value = performance_matrix[i, j]
            text_color = 'white' if value > 50 else 'black'
            text = ax.text(j, i, f'{value:.1f}', 
                          ha="center", va="center", 
                          color=text_color, fontsize=10, fontweight='bold')
    
    # 添加标题
    title = f'Cross-Architecture Attack Heatmap\n({dataset.upper()}, Feature Group: {group})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存至: {save_path}")
    
    # 保存 PDF 版本
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF版本已保存至: {pdf_path}")
    
    plt.show()

def plot_comparison_bar(base_dir='results', dataset='cifar10', group='G4', save_path='cross_arch_comparison.png'):
    """
    绘制同架构 vs 跨架构攻击对比柱状图
    """
    models = ['vgg16', 'resnet50', 'densenet121', 'mobilenetv2']
    model_labels = ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2']
    
    # 收集同架构和跨架构的平均性能
    same_arch_performance = []
    cross_arch_performance = []
    
    for victim in models:
        # 同架构
        same_tpr = load_attack_result(base_dir, dataset, victim, victim, group)
        same_arch_performance.append(same_tpr * 100 if same_tpr else 0)
        
        # 跨架构平均
        cross_tprs = []
        for shadow in models:
            if shadow != victim:
                tpr = load_attack_result(base_dir, dataset, victim, shadow, group)
                if tpr:
                    cross_tprs.append(tpr * 100)
        
        cross_arch_performance.append(np.mean(cross_tprs) if cross_tprs else 0)
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, same_arch_performance, width, 
                   label='Same Architecture', color='#2ca02c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, cross_arch_performance, width, 
                   label='Cross Architecture (Avg)', color='#d62728', alpha=0.8, edgecolor='black')
    
    # 在柱子上添加数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xlabel('Target Model Architecture', fontsize=13, fontweight='bold')
    ax.set_ylabel('TPR@0.1%FPR (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    
    # 保存 PDF 版本
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF版本已保存至: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制跨架构攻击性能图表')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='结果基础目录 (默认: results)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='数据集名称 (默认: cifar10)')
    parser.add_argument('--group', type=str, default='G4',
                       choices=['G0', 'G1', 'G2', 'G3', 'G4'],
                       help='特征组合 (默认: G4)')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['scatter', 'bar', 'all'],
                       help='图表类型 (默认: all)')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='输出目录 (默认: plots)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("=" * 80)
    print("跨架构攻击性能可视化")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"特征组: {args.group}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 生成图表
    if args.plot_type in ['scatter', 'all']:
        scatter_path = os.path.join(args.output_dir, f'cross_arch_attack_{args.dataset}_{args.group}_performance.png')
        plot_cross_arch_matrix(args.base_dir, args.dataset, args.group, scatter_path)
    
    if args.plot_type in ['bar', 'all']:
        bar_path = os.path.join(args.output_dir, f'same_vs_cross_arch_{args.dataset}_{args.group}.png')
        plot_comparison_bar(args.base_dir, args.dataset, args.group, bar_path)
    
    print("\n" + "=" * 80)
    print("所有图表生成完成！")
    print("=" * 80)
