"""
绘制RAPID攻击分数的直方图对比
对比G0(原始)和G4策略的成员/非成员分数分布
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogLocator, NullFormatter

# 设置字体
matplotlib.rcParams.update({'font.size': 14})

def load_attack_scores(folder_path):
    """
    加载攻击结果的分数
    
    Args:
        folder_path: 攻击结果文件夹路径
    
    Returns:
        tuple: (member_scores, non_member_scores)
    """
    npz_file = os.path.join(folder_path, "Roc_confidence_score.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"未找到文件: {npz_file}")
    
    data = np.load(npz_file)
    labels = data['ROC_label']  # 1=member, 0=non-member
    scores = data['ROC_confidence_score']
    
    member_scores = scores[labels == 1]
    non_member_scores = scores[labels == 0]
    
    return member_scores, non_member_scores


def plot_score_distribution(g0_folder, g4_folder, output_path=None):
    """
    绘制G0和G4策略的分数分布对比图
    
    Args:
        g0_folder: G0攻击结果文件夹路径
        g4_folder: G4攻击结果文件夹路径
        output_path: 输出图片路径
    """
    # 加载数据
    print("加载G0数据...")
    g0_member, g0_non_member = load_attack_scores(g0_folder)
    print(f"  Member samples: {len(g0_member)}")
    print(f"  Non-member samples: {len(g0_non_member)}")
    print(f"  Member score range: [{g0_member.min():.4f}, {g0_member.max():.4f}]")
    print(f"  Non-member score range: [{g0_non_member.min():.4f}, {g0_non_member.max():.4f}]")
    
    print("\n加载G4数据...")
    g4_member, g4_non_member = load_attack_scores(g4_folder)
    print(f"  Member samples: {len(g4_member)}")
    print(f"  Non-member samples: {len(g4_non_member)}")
    print(f"  Member score range: [{g4_member.min():.4f}, {g4_member.max():.4f}]")
    print(f"  Non-member score range: [{g4_non_member.min():.4f}, {g4_non_member.max():.4f}]")
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 设置bins (使用相同的bins以便对比)
    bins = 150
    
    # 计算non-member的99.9%分位数 (低FPR参考线位置)
    g0_threshold = np.percentile(g0_non_member, 99.9)
    g4_threshold = np.percentile(g4_non_member, 99.9)
    
    # === 左图: G0 (Original RAPID) ===
    ax1 = axes[0]
    
    # 绘制non-member (淡蓝色, 先画以便member在上层)
    ax1.hist(g0_non_member, bins=bins, density=True, alpha=0.6, 
             color='#87CEEB', label='Non-member', edgecolor='white', linewidth=0.5)
    
    # 绘制member (淡黄色)
    ax1.hist(g0_member, bins=bins, density=True, alpha=0.6, 
             color='#FFD966', label='Member', edgecolor='white', linewidth=0.5)
    
    # 添加低FPR参考线 (99.9%分位数)
    ax1.axvline(g0_threshold, color='black', linestyle='--', linewidth=2, 
                label=f'0.1% FPR threshold ({g0_threshold:.4f})', alpha=0.8)
    
    ax1.set_xlabel('Membership Score', fontsize=16)
    ax1.set_ylabel('Probability Density (log scale)', fontsize=16)
    ax1.set_title('(a) RAPID calibrated scores', fontsize=20, pad=15, fontweight='bold')
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=15)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')
    ax1.set_xlim(0.2, 1)  # 裁掉左侧无关区域
    
    # 设置y轴刻度
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    
    # === 右图: G4 (Enhanced RAPID) ===
    ax2 = axes[1]
    
    # 绘制non-member (淡蓝色)
    ax2.hist(g4_non_member, bins=bins, density=True, alpha=0.6, 
             color='#87CEEB', label='Non-member', edgecolor='white', linewidth=0.5)
    
    # 绘制member (淡黄色)
    ax2.hist(g4_member, bins=bins, density=True, alpha=0.6, 
             color='#FFD966', label='Member', edgecolor='white', linewidth=0.5)
    
    # 添加低FPR参考线 (99.9%分位数)
    ax2.axvline(g4_threshold, color='black', linestyle='--', linewidth=2, 
                label=f'0.1% FPR threshold ({g4_threshold:.4f})', alpha=0.8)
    
    ax2.set_xlabel('Membership Score', fontsize=16)
    ax2.set_ylabel('Probability Density (log scale)', fontsize=16)
    ax2.set_title('(b) Feature-enhanced scores', fontsize=20, pad=15, fontweight='bold')
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=14)
    ax2.legend(loc='upper right', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    ax2.set_xlim(0.2, 1)  # 裁掉左侧无关区域
    
    # 设置y轴刻度
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_dir = os.path.dirname(g4_folder)
        output_path = os.path.join(output_dir, "score_distribution_G0_vs_G4.pdf")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    
    # 同时保存PNG格式
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {png_path}")
    
    # 显示统计信息
    print("\n" + "="*60)
    print("统计信息对比")
    print("="*60)
    
    print("\nG0 (Original RAPID):")
    print(f"  Member mean: {g0_member.mean():.4f}, std: {g0_member.std():.4f}")
    print(f"  Non-member mean: {g0_non_member.mean():.4f}, std: {g0_non_member.std():.4f}")
    print(f"  Separation (mean diff): {g0_member.mean() - g0_non_member.mean():.4f}")
    
    print("\nG4 (Enhanced RAPID):")
    print(f"  Member mean: {g4_member.mean():.4f}, std: {g4_member.std():.4f}")
    print(f"  Non-member mean: {g4_non_member.mean():.4f}, std: {g4_non_member.std():.4f}")
    print(f"  Separation (mean diff): {g4_member.mean() - g4_non_member.mean():.4f}")
    
    print("\n改进效果:")
    print(f"  分离度提升: {(g4_member.mean() - g4_non_member.mean()) - (g0_member.mean() - g0_non_member.mean()):.4f}")
    
    # 计算高分区域 (score > 0.9) 的占比
    print("\n高置信度区域 (score > 0.9):")
    print(f"  G0 - Member: {(g0_member > 0.9).sum() / len(g0_member) * 100:.2f}%")
    print(f"  G0 - Non-member: {(g0_non_member > 0.9).sum() / len(g0_non_member) * 100:.2f}%")
    print(f"  G4 - Member: {(g4_member > 0.9).sum() / len(g4_member) * 100:.2f}%")
    print(f"  G4 - Non-member: {(g4_non_member > 0.9).sum() / len(g4_non_member) * 100:.2f}%")
    
    # 计算0.1% FPR阈值 (99.9%分位数)
    g0_threshold = np.percentile(g0_non_member, 99.9)
    g4_threshold = np.percentile(g4_non_member, 99.9)
    
    print("\n低FPR阈值 (0.1% FPR = 99.9%分位数):")
    print(f"  G0 threshold: {g0_threshold:.4f}")
    print(f"    - Member超过阈值: {(g0_member > g0_threshold).sum() / len(g0_member) * 100:.2f}%")
    print(f"    - Non-member超过阈值: {(g0_non_member > g0_threshold).sum() / len(g0_non_member) * 100:.2f}% (应接近0.1%)")
    print(f"  G4 threshold: {g4_threshold:.4f}")
    print(f"    - Member超过阈值: {(g4_member > g4_threshold).sum() / len(g4_member) * 100:.2f}%")
    print(f"    - Non-member超过阈值: {(g4_non_member > g4_threshold).sum() / len(g4_non_member) * 100:.2f}% (应接近0.1%)")
    
    print("\nTPR@0.1%FPR提升:")
    tpr_g0 = (g0_member > g0_threshold).sum() / len(g0_member) * 100
    tpr_g4 = (g4_member > g4_threshold).sum() / len(g4_member) * 100
    print(f"  G0: {tpr_g0:.2f}%")
    print(f"  G4: {tpr_g4:.2f}%")
    print(f"  提升: +{tpr_g4 - tpr_g0:.2f}%")
    print("="*60)
    
    print("\nTPR@0.01%FPR提升:")
    tpr_g0 = (g0_member > g0_threshold).sum() / len(g0_member) * 100
    tpr_g4 = (g4_member > g4_threshold).sum() / len(g4_member) * 100
    print(f"  G0: {tpr_g0:.2f}%")
    print(f"  G4: {tpr_g4:.2f}%")
    print(f"  提升: +{tpr_g4 - tpr_g0:.2f}%")
    print("="*60)


if __name__ == '__main__':
    # 数据来源路径
    g0_folder = r"D:\Users\Administrator\PycharmProjects\RAPID-main服务器\RAPID-main\results\cifar10_resnet50\RAPID\rapid_attack_G0"
    g4_folder = r"D:\Users\Administrator\PycharmProjects\RAPID-main服务器\RAPID-main\results\cifar10_resnet50\RAPID\rapid_attack_G4"
    
    print("="*60)
    print("RAPID攻击分数分布对比 (G0 vs G4)")
    print("="*60)
    print(f"G0数据来源: {g0_folder}")
    print(f"G4数据来源: {g4_folder}")
    print("="*60)
    print()
    
    # 绘制对比图
    plot_score_distribution(g0_folder, g4_folder)
