"""
从保存的攻击结果文件生成详细的日志报告
"""
import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve
from datetime import datetime
import argparse

def analyze_attack_results(npz_path, output_log_path=None):
    """
    分析攻击结果并生成详细日志
    
    参数:
        npz_path: Roc_confidence_score.npz 文件路径
        output_log_path: 输出日志文件路径(可选)
    """
    # 加载结果数据
    data = np.load(npz_path)
    ROC_label = data['ROC_label']
    ROC_confidence_score = data['ROC_confidence_score']
    
    # 处理NaN值
    ROC_confidence_score = np.nan_to_num(ROC_confidence_score, nan=np.nanmean(ROC_confidence_score))
    
    # 计算各项指标
    fpr, tpr, thresholds = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # 计算不同FPR阈值下的TPR
    tpr_at_fpr = {}
    fpr_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    for fpr_threshold in fpr_thresholds:
        idx = np.where(fpr < fpr_threshold)[0]
        if len(idx) > 0:
            tpr_at_fpr[fpr_threshold] = tpr[idx[-1]]
        else:
            tpr_at_fpr[fpr_threshold] = 0.0
    
    # 计算准确率(使用最优阈值)
    # Youden's J statistic (TPR - FPR)
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]
    
    predictions = (ROC_confidence_score >= best_threshold).astype(int)
    accuracy = accuracy_score(ROC_label, predictions)
    
    # 计算Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(ROC_label, ROC_confidence_score)
    pr_auc = auc(recall, precision)
    
    # 成员/非成员分数统计
    member_scores = ROC_confidence_score[ROC_label == 1]
    non_member_scores = ROC_confidence_score[ROC_label == 0]
    
    # 构建日志内容
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("RAPID 成员推断攻击结果报告")
    log_lines.append("=" * 80)
    log_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"结果文件: {npz_path}")
    log_lines.append("")
    
    log_lines.append("-" * 80)
    log_lines.append("核心性能指标")
    log_lines.append("-" * 80)
    log_lines.append(f"ROC AUC:           {roc_auc:.4f}")
    log_lines.append(f"PR AUC:            {pr_auc:.4f}")
    log_lines.append(f"准确率(Accuracy):  {accuracy*100:.2f}%")
    log_lines.append(f"最优阈值:          {best_threshold:.4f}")
    log_lines.append("")
    
    log_lines.append("-" * 80)
    log_lines.append("不同FPR阈值下的TPR")
    log_lines.append("-" * 80)
    for fpr_threshold in fpr_thresholds:
        tpr_value = tpr_at_fpr[fpr_threshold]
        log_lines.append(f"TPR @ {fpr_threshold*100:>5.2f}% FPR: {tpr_value*100:6.2f}%")
    log_lines.append("")
    
    log_lines.append("-" * 80)
    log_lines.append("样本统计")
    log_lines.append("-" * 80)
    log_lines.append(f"总样本数:          {len(ROC_label)}")
    log_lines.append(f"成员样本数:        {np.sum(ROC_label == 1)}")
    log_lines.append(f"非成员样本数:      {np.sum(ROC_label == 0)}")
    log_lines.append("")
    
    log_lines.append("-" * 80)
    log_lines.append("成员分数统计")
    log_lines.append("-" * 80)
    log_lines.append(f"均值:              {member_scores.mean():.4f}")
    log_lines.append(f"标准差:            {member_scores.std():.4f}")
    log_lines.append(f"中位数:            {np.median(member_scores):.4f}")
    log_lines.append(f"最小值:            {member_scores.min():.4f}")
    log_lines.append(f"最大值:            {member_scores.max():.4f}")
    log_lines.append("")
    
    log_lines.append("-" * 80)
    log_lines.append("非成员分数统计")
    log_lines.append("-" * 80)
    log_lines.append(f"均值:              {non_member_scores.mean():.4f}")
    log_lines.append(f"标准差:            {non_member_scores.std():.4f}")
    log_lines.append(f"中位数:            {np.median(non_member_scores):.4f}")
    log_lines.append(f"最小值:            {non_member_scores.min():.4f}")
    log_lines.append(f"最大值:            {non_member_scores.max():.4f}")
    log_lines.append("")
    
    log_lines.append("-" * 80)
    log_lines.append("分数分布分析")
    log_lines.append("-" * 80)
    log_lines.append(f"成员-非成员均值差: {member_scores.mean() - non_member_scores.mean():.4f}")
    log_lines.append(f"分离度(Cohen's d): {(member_scores.mean() - non_member_scores.mean()) / np.sqrt((member_scores.std()**2 + non_member_scores.std()**2) / 2):.4f}")
    log_lines.append("")
    
    log_lines.append("=" * 80)
    
    # 打印到控制台
    log_content = "\n".join(log_lines)
    print(log_content)
    
    # 保存到文件
    if output_log_path:
        with open(output_log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        print(f"\n日志已保存到: {output_log_path}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'tpr_at_fpr': tpr_at_fpr,
        'best_threshold': best_threshold
    }

def batch_analyze_all_datasets(base_path, analyze_all=False, ablation_mode=False):
    """
    批量分析所有数据集-模型组合的攻击结果
    
    参数:
        base_path: 基础路径
        analyze_all: 是否分析所有攻击类型,默认False只分析核心类型
        ablation_mode: 是否只分析消融实验结果(带m{num}_q{num}后缀),默认False
    """
    results_base = os.path.join(base_path, 'results')
    
    if not os.path.exists(results_base):
        print(f"错误: 未找到 results 文件夹: {results_base}")
        return
    
    # 扫描所有数据集-模型文件夹
    dataset_models = []
    for item in os.listdir(results_base):
        item_path = os.path.join(results_base, item)
        if os.path.isdir(item_path):
            rapid_path = os.path.join(item_path, 'RAPID')
            if os.path.exists(rapid_path) and os.path.isdir(rapid_path):
                dataset_models.append(item)
    
    if not dataset_models:
        print(f"未找到任何包含 RAPID 文件夹的数据集-模型组合")
        return
    
    if ablation_mode:
        attack_mode = "消融实验(rapid_attack_G*_m*_q*)"
    elif analyze_all:
        attack_mode = "所有"
    else:
        attack_mode = "核心(rapid_attack, G0-G4)"
    
    print("=" * 120)
    print(f"找到 {len(dataset_models)} 个数据集-模型组合 (分析模式: {attack_mode}):")
    for dm in sorted(dataset_models):
        print(f"  - {dm}")
    print("=" * 120)
    print()
    
    # 逐个分析
    all_summaries = []
    for dataset_model in sorted(dataset_models):
        print(f"\n{'#' * 120}")
        print(f"# 开始分析: {dataset_model}")
        print(f"{'#' * 120}\n")
        
        results = batch_analyze_attacks(base_path, dataset_model, analyze_all=analyze_all, ablation_mode=ablation_mode)
        
        if results:
            all_summaries.append({
                'name': dataset_model,
                'results': results
            })
    
    # 生成全局汇总
    print(f"\n\n{'=' * 120}")
    print("全局汇总 - 所有数据集-模型的最佳结果")
    print("=" * 120)
    
    global_summary_lines = []
    global_summary_lines.append("=" * 140)
    global_summary_lines.append("全局汇总 - 所有数据集-模型的最佳攻击配置")
    global_summary_lines.append("=" * 140)
    global_summary_lines.append(f"{'数据集-模型':<35} {'最佳配置':<30} {'ROC AUC':<10} {'Acc%':<10} {'TPR@0.01%':<12}")
    global_summary_lines.append("-" * 140)
    
    for summary in all_summaries:
        dataset_model = summary['name']
        results = summary['results']
        
        # 找到最佳配置(基于ROC AUC)
        best_config = max(results.items(), key=lambda x: x[1]['roc_auc'])
        config_name = best_config[0]
        metrics = best_config[1]
        
        global_summary_lines.append(
            f"{dataset_model:<35} {config_name:<30} {metrics['roc_auc']:<10.4f} "
            f"{metrics['accuracy']*100:<9.2f}% {metrics['tpr_at_fpr'].get(0.0001, 0)*100:<11.2f}%"
        )
    
    global_summary_lines.append("=" * 140)
    
    global_summary_content = "\n".join(global_summary_lines)
    print(global_summary_content)
    
    # 保存全局汇总
    global_summary_path = os.path.join(results_base, 'global_attack_summary.txt')
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        f.write(global_summary_content)
    print(f"\n全局汇总已保存到: {global_summary_path}")
    
    return all_summaries

def batch_analyze_attacks(base_path, dataset_model_name, attack_types=None, analyze_all=False, ablation_mode=False):
    """
    批量分析单个数据集-模型的多个攻击结果
    
    参数:
        base_path: 基础路径
        dataset_model_name: 数据集_模型名称,如 'cifar10_densenet121'
        attack_types: 攻击类型列表,如 ['rapid_attack_G0', 'rapid_attack_G1']
        analyze_all: 是否分析所有攻击类型,默认False只分析核心类型
        ablation_mode: 是否只分析消融实验结果(带m{num}_q{num}后缀),默认False
    """
    results_path = os.path.join(base_path, 'results', dataset_model_name, 'RAPID')
    
    if attack_types is None:
        if ablation_mode:
            # 消融模式: 只分析带有 _r{num}_q{num} 或 _m{num}_q{num} 后缀的攻击
            import re
            ablation_pattern = re.compile(r'rapid_attack_G[0-4]_[rm]\d+_q\d+$')
            attack_types = [d for d in os.listdir(results_path) 
                           if os.path.isdir(os.path.join(results_path, d)) and ablation_pattern.match(d)]
        elif analyze_all:
            # 扫描所有攻击类型
            attack_types = [d for d in os.listdir(results_path) 
                           if os.path.isdir(os.path.join(results_path, d)) and d.startswith('rapid_attack')]
        else:
            # 默认只分析核心攻击类型
            core_attack_types = ['rapid_attack', 'rapid_attack_G0', 'rapid_attack_G1', 
                                'rapid_attack_G2', 'rapid_attack_G3', 'rapid_attack_G4']
            # 检查哪些类型实际存在
            attack_types = [d for d in core_attack_types 
                           if os.path.exists(os.path.join(results_path, d))]
    
    all_results = {}
    summary_lines = []
    
    summary_lines.append("=" * 130)
    summary_lines.append(f"{dataset_model_name} - RAPID攻击结果汇总")
    summary_lines.append("=" * 130)
    summary_lines.append(f"{'攻击类型':<30} {'ROC AUC':<10} {'Accuracy':<12} {'TPR@0.01%':<12} {'TPR@0.05%':<12} {'TPR@0.1%':<12} {'TPR@1%':<12}")
    summary_lines.append("-" * 130)
    
    # 如果是消融模式，额外创建详细对比表
    detailed_lines = []
    if ablation_mode:
        detailed_lines.append("=" * 150)
        detailed_lines.append(f"{dataset_model_name} - 消融实验详细对比")
        detailed_lines.append("=" * 150)
        cohens_d_header = "Cohen's d"
        detailed_lines.append(f"{'攻击配置':<25} {'ROC AUC':<10} {'PR AUC':<10} {'Accuracy':<10} "
                             f"{'TPR@0.01%':<11} {'TPR@0.1%':<11} {'TPR@1%':<11} {'最优阈值':<10} {cohens_d_header:<10}")
        detailed_lines.append("-" * 150)
    
    # 自定义排序函数：特征组 -> 模型数 -> 查询数
    def sort_attack_types(attack_name):
        import re
        # 提取 G数字, r/m数字, q数字
        match = re.search(r'G(\d+)_[rm](\d+)_q(\d+)', attack_name)
        if match:
            g_num = int(match.group(1))  # 特征组编号
            model_num = int(match.group(2))  # 模型数量
            query_num = int(match.group(3))  # 查询数量
            return (g_num, model_num, query_num)
        return (999, 999, 999)  # 无法解析的放最后
    
    for attack_type in sorted(attack_types, key=sort_attack_types):
        npz_path = os.path.join(results_path, attack_type, 'Roc_confidence_score.npz')
        log_path = os.path.join(results_path, attack_type, 'attack_log.txt')
        
        if not os.path.exists(npz_path):
            print(f"跳过 {attack_type}: 未找到结果文件")
            continue
        
        print(f"\n{'='*50}")
        print(f"分析 {attack_type}")
        print(f"{'='*50}")
        
        results = analyze_attack_results(npz_path, log_path)
        all_results[attack_type] = results
        
        # 添加到汇总
        tpr_0001 = results['tpr_at_fpr'].get(0.0001, 0) * 100
        tpr_0005 = results['tpr_at_fpr'].get(0.0005, 0) * 100
        tpr_001 = results['tpr_at_fpr'].get(0.001, 0) * 100
        tpr_01 = results['tpr_at_fpr'].get(0.01, 0) * 100
        summary_lines.append(
            f"{attack_type:<30} {results['roc_auc']:<10.4f} {results['accuracy']*100:<11.2f}% "
            f"{tpr_0001:<11.2f}% {tpr_0005:<11.2f}% {tpr_001:<11.2f}% {tpr_01:<11.2f}%"
        )
        
        # 如果是消融模式，添加到详细对比表
        if ablation_mode:
            # 提取参数配置 (例如 "G0_r4_q8")
            import re
            match = re.search(r'(G[0-4]_[rm]\d+_q\d+)', attack_type)
            config_str = match.group(1) if match else attack_type
            
            # 加载完整数据以计算Cohen's d
            data = np.load(npz_path)
            ROC_label = data['ROC_label']
            ROC_confidence_score = data['ROC_confidence_score']
            ROC_confidence_score = np.nan_to_num(ROC_confidence_score, nan=np.nanmean(ROC_confidence_score))
            
            member_scores = ROC_confidence_score[ROC_label == 1]
            non_member_scores = ROC_confidence_score[ROC_label == 0]
            cohens_d = (member_scores.mean() - non_member_scores.mean()) / np.sqrt((member_scores.std()**2 + non_member_scores.std()**2) / 2)
            
            detailed_lines.append(
                f"{config_str:<25} {results['roc_auc']:<10.4f} {results['pr_auc']:<10.4f} {results['accuracy']*100:<9.2f}% "
                f"{tpr_0001:<10.2f}% {tpr_001:<10.2f}% {tpr_01:<10.2f}% {results['best_threshold']:<10.4f} {cohens_d:<10.4f}"
            )
    
    summary_lines.append("=" * 130)
    
    # 保存汇总报告
    summary_content = "\n".join(summary_lines)
    print(f"\n\n{summary_content}")
    
    summary_filename = f'attack_summary_{dataset_model_name}_ablation.txt' if ablation_mode else f'attack_summary_{dataset_model_name}.txt'
    summary_path = os.path.join(results_path, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    print(f"\n汇总报告已保存到: {summary_path}")
    
    # 如果是消融模式，保存详细对比表
    if ablation_mode and detailed_lines:
        detailed_lines.append("=" * 150)
        detailed_content = "\n".join(detailed_lines)
        print(f"\n\n{detailed_content}")
        
        detailed_path = os.path.join(results_path, f'ablation_detailed_{dataset_model_name}.txt')
        with open(detailed_path, 'w', encoding='utf-8') as f:
            f.write(detailed_content)
        print(f"\n消融实验详细对比表已保存到: {detailed_path}")
    
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从保存的文件生成RAPID攻击结果日志')
    parser.add_argument('--base_path', type=str, 
                       default=r'D:\Users\Administrator\PycharmProjects\RAPID-main服务器\RAPID-main',
                       help='项目基础路径')
    parser.add_argument('--dataset_model', type=str, default=None,
                       help='数据集_模型名称,如 cifar10_densenet121。留空则分析所有数据集-模型')
    parser.add_argument('--attack_type', type=str, default=None,
                       help='指定单个攻击类型,如 rapid_attack_G0。留空则分析所有类型')
    parser.add_argument('--batch', action='store_true',
                       help='批量分析所有攻击类型')
    parser.add_argument('--all_attacks', action='store_true',
                       help='分析所有攻击类型(包括实验性类型),默认只分析核心类型(rapid_attack, G0-G4)')
    parser.add_argument('--ablation', action='store_true',
                       help='消融模式: 只分析带有参数后缀的攻击(如rapid_attack_G0_m4_q8)')
    
    args = parser.parse_args()
    
    if args.dataset_model is None and args.attack_type is None:
        # 分析所有数据集-模型
        batch_analyze_all_datasets(args.base_path, analyze_all=args.all_attacks, ablation_mode=args.ablation)
    elif args.batch or args.attack_type is None:
        # 批量分析指定数据集-模型的所有攻击类型
        if args.dataset_model is None:
            print("错误: 请指定 --dataset_model 或直接运行以分析所有数据集")
        else:
            batch_analyze_attacks(args.base_path, args.dataset_model, analyze_all=args.all_attacks, ablation_mode=args.ablation)
        batch_analyze_all_datasets(args.base_path, analyze_all=args.all_attacks)
    elif args.batch or args.attack_type is None:
        # 批量分析指定数据集-模型的所有攻击类型
        if args.dataset_model is None:
            print("错误: 请指定 --dataset_model 或直接运行以分析所有数据集")
        else:
            batch_analyze_attacks(args.base_path, args.dataset_model, analyze_all=args.all_attacks)
    else:
        # 单个分析
        if args.dataset_model is None:
            print("错误: 请指定 --dataset_model")
        else:
            results_path = os.path.join(args.base_path, 'results', args.dataset_model, 'RAPID')
            npz_path = os.path.join(results_path, args.attack_type, 'Roc_confidence_score.npz')
            log_path = os.path.join(results_path, args.attack_type, 'attack_log.txt')
            
            if os.path.exists(npz_path):
                analyze_attack_results(npz_path, log_path)
            else:
                print(f"错误: 未找到文件 {npz_path}")
