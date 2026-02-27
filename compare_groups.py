"""
对比 G0-G4 策略的攻击效果
默认对比 r=4, q=8 配置下的 G0-G4 策略
"""
import os
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc

def find_result_file(base_path, group, ref_model=4, query_num=8):
    """
    查找攻击结果文件，支持自动回退到基线版本
    
    Args:
        base_path: RAPID文件夹路径
        group: 特征组 (G0-G4)
        ref_model: 参考模型数量
        query_num: 查询次数
    
    Returns:
        str: 结果文件路径，如果未找到返回 None
    """
    # 首先尝试精确匹配
    folder_name = f"rapid_attack_{group}_r{ref_model}_q{query_num}"
    result_path = os.path.join(base_path, folder_name, "Roc_confidence_score.npz")
    
    if os.path.exists(result_path):
        return result_path
    
    # 如果找不到，回退到基线版本 (rapid_attack_G0, rapid_attack_G1 等)
    baseline_folder = f"rapid_attack_{group}"
    baseline_path = os.path.join(base_path, baseline_folder, "Roc_confidence_score.npz")
    
    if os.path.exists(baseline_path):
        return baseline_path
    
    return None

def analyze_result(npz_path):
    """
    分析单个攻击结果
    
    Returns:
        dict: 包含各项指标的字典
    """
    data = np.load(npz_path)
    ROC_label = data['ROC_label']
    ROC_confidence_score = data['ROC_confidence_score']
    
    # 处理NaN
    ROC_confidence_score = np.nan_to_num(ROC_confidence_score, nan=np.nanmean(ROC_confidence_score))
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # 计算不同FPR下的TPR
    tpr_at_fpr = {}
    for fpr_threshold in [0.0001, 0.001, 0.01]:
        idx = np.where(fpr < fpr_threshold)[0]
        tpr_at_fpr[fpr_threshold] = tpr[idx[-1]] if len(idx) > 0 else 0.0
    
    # 准确率
    acc = data.get('acc', 0.0)
    
    return {
        'roc_auc': roc_auc,
        'accuracy': acc,
        'tpr_0001': tpr_at_fpr[0.0001],
        'tpr_001': tpr_at_fpr[0.001],
        'tpr_01': tpr_at_fpr[0.01]
    }

def compare_groups(dataset_model, ref_model=4, query_num=8, base_dir='results'):
    """
    对比 G0-G4 策略的效果
    
    Args:
        dataset_model: 数据集_模型名称
        ref_model: 参考模型数量
        query_num: 查询次数
        base_dir: 结果基础目录
    """
    rapid_path = os.path.join(base_dir, dataset_model, 'RAPID')
    
    if not os.path.exists(rapid_path):
        print(f"错误: 未找到 {rapid_path}")
        return
    
    print("=" * 100)
    print(f"{dataset_model} - G0-G4 策略对比 (r={ref_model}, q={query_num})")
    print("=" * 100)
    print(f"{'策略':<10} {'ROC AUC':<10} {'Accuracy':<12} {'TPR@0.01%':<12} {'TPR@0.1%':<12} {'TPR@1%':<12}")
    print("-" * 100)
    
    results = {}
    for group in ['G0', 'G1', 'G2', 'G3', 'G4']:
        npz_path = find_result_file(rapid_path, group, ref_model, query_num)
        
        if npz_path is None:
            print(f"{group:<10} 未找到结果文件")
            continue
        
        try:
            metrics = analyze_result(npz_path)
            results[group] = metrics
            
            print(f"{group:<10} {metrics['roc_auc']:<10.4f} {metrics['accuracy']:<11.2f}% "
                  f"{metrics['tpr_0001']*100:<11.2f}% {metrics['tpr_001']*100:<11.2f}% {metrics['tpr_01']*100:<11.2f}%")
        except Exception as e:
            print(f"{group:<10} 加载失败: {e}")
    
    print("=" * 100)
    
    # 找出最佳策略
    if results:
        best_group = max(results.items(), key=lambda x: x[1]['roc_auc'])
        print(f"\n最佳策略: {best_group[0]} (ROC AUC = {best_group[1]['roc_auc']:.4f})")
    
    # 保存对比结果
    save_comparison(dataset_model, results, ref_model, query_num, base_dir)
    
    return results

def save_comparison(dataset_model, results, ref_model, query_num, base_dir):
    """保存对比结果到文件"""
    output_path = os.path.join(base_dir, dataset_model, 'RAPID', 
                               f'groups_comparison_r{ref_model}_q{query_num}.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"{dataset_model} - G0-G4 策略对比 (r={ref_model}, q={query_num})\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'策略':<10} {'ROC AUC':<10} {'Accuracy':<12} {'TPR@0.01%':<12} {'TPR@0.1%':<12} {'TPR@1%':<12}\n")
        f.write("-" * 100 + "\n")
        
        for group in ['G0', 'G1', 'G2', 'G3', 'G4']:
            if group in results:
                m = results[group]
                f.write(f"{group:<10} {m['roc_auc']:<10.4f} {m['accuracy']:<11.2f}% "
                       f"{m['tpr_0001']*100:<11.2f}% {m['tpr_001']*100:<11.2f}% {m['tpr_01']*100:<11.2f}%\n")
        
        f.write("=" * 100 + "\n")
        
        if results:
            best_group = max(results.items(), key=lambda x: x[1]['roc_auc'])
            f.write(f"\n最佳策略: {best_group[0]} (ROC AUC = {best_group[1]['roc_auc']:.4f})\n")
    
    print(f"\n对比结果已保存至: {output_path}")

def batch_compare_all(base_dir='results', ref_model=4, query_num=8):
    """批量对比所有数据集-模型组合"""
    if not os.path.exists(base_dir):
        print(f"错误: 未找到目录 {base_dir}")
        return
    
    # 扫描所有数据集-模型组合
    dataset_models = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            rapid_path = os.path.join(item_path, 'RAPID')
            if os.path.exists(rapid_path):
                dataset_models.append(item)
    
    if not dataset_models:
        print("未找到任何数据集-模型组合")
        return
    
    print(f"\n找到 {len(dataset_models)} 个数据集-模型组合\n")
    
    # 汇总所有结果
    all_results = {}
    for dataset_model in sorted(dataset_models):
        results = compare_groups(dataset_model, ref_model, query_num, base_dir)
        if results:
            all_results[dataset_model] = results
        print()
    
    # 生成全局汇总
    print("\n" + "=" * 120)
    print("全局汇总 - 所有数据集-模型的最佳策略")
    print("=" * 120)
    print(f"{'数据集-模型':<40} {'最佳策略':<10} {'ROC AUC':<10} {'Accuracy':<12} {'TPR@0.01%':<12}")
    print("-" * 120)
    
    for dataset_model, results in sorted(all_results.items()):
        if results:
            best_group = max(results.items(), key=lambda x: x[1]['roc_auc'])
            m = best_group[1]
            print(f"{dataset_model:<40} {best_group[0]:<10} {m['roc_auc']:<10.4f} "
                  f"{m['accuracy']:<11.2f}% {m['tpr_0001']*100:<11.2f}%")
    
    print("=" * 120)
    
    # 保存全局汇总
    global_summary_path = os.path.join(base_dir, f'global_groups_comparison_r{ref_model}_q{query_num}.txt')
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("全局汇总 - 所有数据集-模型的最佳策略\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'数据集-模型':<40} {'最佳策略':<10} {'ROC AUC':<10} {'Accuracy':<12} {'TPR@0.01%':<12}\n")
        f.write("-" * 120 + "\n")
        
        for dataset_model, results in sorted(all_results.items()):
            if results:
                best_group = max(results.items(), key=lambda x: x[1]['roc_auc'])
                m = best_group[1]
                f.write(f"{dataset_model:<40} {best_group[0]:<10} {m['roc_auc']:<10.4f} "
                       f"{m['accuracy']:<11.2f}% {m['tpr_0001']*100:<11.2f}%\n")
        
        f.write("=" * 120 + "\n")
    
    print(f"\n全局汇总已保存至: {global_summary_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='对比 G0-G4 策略的攻击效果')
    parser.add_argument('--dataset_model', type=str, default=None,
                       help='数据集_模型名称，如 cifar10_vgg16。留空则分析所有')
    parser.add_argument('--ref_model', type=int, default=4,
                       help='参考模型数量 (默认: 4)')
    parser.add_argument('--query_num', type=int, default=8,
                       help='查询次数 (默认: 8)')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='结果基础目录 (默认: results)')
    
    args = parser.parse_args()
    
    if args.dataset_model:
        # 分析单个数据集-模型
        compare_groups(args.dataset_model, args.ref_model, args.query_num, args.base_dir)
    else:
        # 批量分析所有
        batch_compare_all(args.base_dir, args.ref_model, args.query_num)
