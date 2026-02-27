import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.base_model import BaseModel
from utils.utils import seed_worker
from tqdm import tqdm
from utils.utils import roc_plot
from sklearn.metrics import roc_curve
import numpy as np

class MiaAttack:
    def __init__(self, victim_model, victim_train_loader, victim_test_loader,
                 shadow_model, shadow_train_loader, shadow_test_loader,
                 verify_victim_model_list=[], verify_shadow_model_list=[],
                 device="cuda", num_cls=10, lr=0.001, weight_decay=5e-4,
                 optimizer="adam", scheduler="", epochs=50, batch_size=128, 
                 dataset_name="cifar10", model_name="vgg16", query_num=1,
                 feature_group="G0", attack_model_type="mia_fc"):
        self.victim_model = victim_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model = shadow_model
        self.shadow_train_loader = shadow_train_loader
        self.shadow_test_loader = shadow_test_loader
        self.device = device
        self.num_cls = num_cls
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.query_num = query_num
        self.feature_group = feature_group
        self.attack_model_type = attack_model_type
        self.rapid_victim_model_list = verify_victim_model_list
        self.rapid_shadow_model_list = verify_shadow_model_list
        self._prepare_rapid()
    
    def _prepare_rapid(self):
        victim_in_confidence_list = []
        victim_out_confidence_list = []
        shadow_in_confidence_list = []
        shadow_out_confidence_list = []
        
        for idx in tqdm(range(self.query_num)):
            victim_in_confidences, victim_in_targets = self.victim_model.predict_target_loss(self.victim_train_loader)
            victim_out_confidences, victim_out_targets = self.victim_model.predict_target_loss(self.victim_test_loader)
            victim_in_confidence_list.append(victim_in_confidences)
            victim_out_confidence_list.append(victim_out_confidences)

            attack_in_confidences, attack_in_targets = self.shadow_model.predict_target_loss(self.shadow_train_loader)
            attack_out_confidences, attack_out_targets = self.shadow_model.predict_target_loss(self.shadow_test_loader)
            shadow_in_confidence_list.append(attack_in_confidences)
            shadow_out_confidence_list.append(attack_out_confidences)
            
        self.attack_in_confidences, self.attack_in_targets = \
            torch.cat(shadow_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True), attack_in_targets
        self.attack_out_confidences, self.attack_out_targetss = \
            torch.cat(shadow_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True), attack_out_targets
        
        self.victim_in_confidences, self.victim_in_targets = \
            torch.cat(victim_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True), victim_in_targets
        self.victim_out_confidences, self.victim_out_targets = \
            torch.cat(victim_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True), victim_out_targets
            
        # 处理参考模型列表:
        # 如果传入的是原始列表(每个元素是一个模型的输出),则拼接成矩阵
        # 如果传入的已经是聚合后的结果,则直接使用
        if len(self.rapid_shadow_model_list) > 0:
            if isinstance(self.rapid_shadow_model_list[0], list) and len(self.rapid_shadow_model_list[0]) > 0:
                # 原始列表: 每个元素是一个参考模型的输出
                self.rapid_attack_in_confidences_raw = torch.cat(self.rapid_shadow_model_list[0], dim=1)
                self.rapid_attack_out_confidences_raw = torch.cat(self.rapid_shadow_model_list[1], dim=1)
                self.rapid_victim_in_confidences_raw = torch.cat(self.rapid_victim_model_list[0], dim=1)
                self.rapid_victim_out_confidences_raw = torch.cat(self.rapid_victim_model_list[1], dim=1)
            else:
                # 已经聚合的结果
                self.rapid_attack_in_confidences_raw = self.rapid_shadow_model_list[0]
                self.rapid_attack_out_confidences_raw = self.rapid_shadow_model_list[1]
                self.rapid_victim_in_confidences_raw = self.rapid_victim_model_list[0]
                self.rapid_victim_out_confidences_raw = self.rapid_victim_model_list[1]
        else:
            # 没有参考模型,创建空张量
            self.rapid_attack_in_confidences_raw = torch.zeros_like(self.attack_in_confidences)
            self.rapid_attack_out_confidences_raw = torch.zeros_like(self.attack_out_confidences)
            self.rapid_victim_in_confidences_raw = torch.zeros_like(self.victim_in_confidences)
            self.rapid_victim_out_confidences_raw = torch.zeros_like(self.victim_out_confidences)
            
        # 计算平均值(用于G0等组合)
        self.rapid_attack_in_confidences = self.rapid_attack_in_confidences_raw.mean(dim=1, keepdim=True)
        self.rapid_attack_out_confidences = self.rapid_attack_out_confidences_raw.mean(dim=1, keepdim=True)
        self.rapid_victim_in_confidences = self.rapid_victim_in_confidences_raw.mean(dim=1, keepdim=True)
        self.rapid_victim_out_confidences = self.rapid_victim_out_confidences_raw.mean(dim=1, keepdim=True)

    def _compute_enhanced_features(self, S, ref_scores_raw):
        """
        计算增强特征
        
        参数:
            S: 初始成员分数 (batch_size, 1)
            ref_scores_raw: 参考模型分数矩阵 (batch_size, num_ref_models)
        
        返回:
            features_dict: 包含所有7个新特征的字典
        """
        eps = 1e-8  # 避免除零的小常数
        
        # (1) mean_ref: 参考模型平均分数
        mean_ref = ref_scores_raw.mean(dim=1, keepdim=True)
        
        # (2) var_ref: 参考模型分数方差
        var_ref = ref_scores_raw.var(dim=1, keepdim=True, unbiased=False)
        
        # (3) z_score: 标准化分数
        std_ref = torch.sqrt(var_ref + eps)
        z_score = (S - mean_ref) / std_ref
        
        # (4) ratio_rho: 比例偏差
        ratio_rho = S / (mean_ref + eps)
        
        # (5) log_rho: 比例偏差的对数变换
        log_rho = torch.log1p(ratio_rho)
        
        # S_prime 已经在调用处计算了 (S - mean_ref)
        
        features_dict = {
            'mean_ref': mean_ref,
            'var_ref': var_ref,
            'z_score': z_score,
            'ratio_rho': ratio_rho,
            'log_rho': log_rho
        }
        
        return features_dict
    
    def _build_feature_combination(self, S, S_prime, ref_scores_raw):
        """
        根据 feature_group 构建特征组合
        
        参数:
            S: 初始成员分数 (batch_size, 1)
            S_prime: 难度校准分数 (S - mean_ref) (batch_size, 1)
            ref_scores_raw: 参考模型分数矩阵 (batch_size, num_ref_models)
        
        返回:
            feature_tensor: 拼接后的特征张量
        """
        eps = 1e-8
        
        if self.feature_group == 'G0':
            # G0: 只包含 S 和 S_prime
            features = [S, S_prime]
        
        elif self.feature_group == 'G1':
            # G1: S, S_prime, mean_ref, var_ref
            enhanced_features = self._compute_enhanced_features(S, ref_scores_raw)
            features = [S, S_prime, enhanced_features['mean_ref'], enhanced_features['var_ref']]
        
        elif self.feature_group == 'G2':
            # G2: S, S_prime, mean_ref, var_ref, z_score
            enhanced_features = self._compute_enhanced_features(S, ref_scores_raw)
            features = [S, S_prime, enhanced_features['mean_ref'], enhanced_features['var_ref'], 
                       enhanced_features['z_score']]
        
        elif self.feature_group == 'G3':
            # G3: S, S_prime, mean_ref, var_ref, z_score, ratio_rho, log_rho
            enhanced_features = self._compute_enhanced_features(S, ref_scores_raw)
            features = [S, S_prime, enhanced_features['mean_ref'], enhanced_features['var_ref'], 
                       enhanced_features['z_score'], enhanced_features['ratio_rho'], enhanced_features['log_rho']]
        
        elif self.feature_group == 'G4':
            # G4: 完整特征集
            # S, S_prime, mean_ref, var_ref, z_score, ratio_rho, log_rho, sp_rho, h
            enhanced_features = self._compute_enhanced_features(S, ref_scores_raw)
            mean_ref = enhanced_features['mean_ref']
            ratio_rho = enhanced_features['ratio_rho']
            
            # (6) sp_rho: 交互特征 S_prime * ratio_rho
            sp_rho = S_prime * ratio_rho
            
            # (7) h: 对称归一化特征
            h = (2 * S_prime * ratio_rho) / (S + mean_ref + eps)
            
            features = [S, S_prime, enhanced_features['mean_ref'], enhanced_features['var_ref'], 
                       enhanced_features['z_score'], enhanced_features['ratio_rho'], enhanced_features['log_rho'],
                       sp_rho, h]
        
        else:
            raise ValueError(f"Unknown feature_group: {self.feature_group}")
        
        # 拼接所有特征
        feature_tensor = torch.cat(features, dim=1)
        return feature_tensor

    def rapid_attack(self, model_name="mia_fc"):
        
        # 为影子模型(训练攻击模型)构建特征
        attack_S = self.attack_in_confidences  # 影子成员的初始分数
        attack_S_prime = attack_S - self.rapid_attack_in_confidences  # 难度校准分数
        attack_in_features = self._build_feature_combination(
            attack_S, attack_S_prime, self.rapid_attack_in_confidences_raw
        )
        
        attack_S_out = self.attack_out_confidences  # 影子非成员的初始分数
        attack_S_prime_out = attack_S_out - self.rapid_attack_out_confidences  # 难度校准分数
        attack_out_features = self._build_feature_combination(
            attack_S_out, attack_S_prime_out, self.rapid_attack_out_confidences_raw
        )
        
        # 为受害者模型(测试攻击模型)构建特征
        victim_S = self.victim_in_confidences  # 受害者成员的初始分数
        victim_S_prime = victim_S - self.rapid_victim_in_confidences  # 难度校准分数
        victim_in_features = self._build_feature_combination(
            victim_S, victim_S_prime, self.rapid_victim_in_confidences_raw
        )
        
        victim_S_out = self.victim_out_confidences  # 受害者非成员的初始分数
        victim_S_prime_out = victim_S_out - self.rapid_victim_out_confidences  # 难度校准分数
        victim_out_features = self._build_feature_combination(
            victim_S_out, victim_S_prime_out, self.rapid_victim_out_confidences_raw
        )
        
        # 合并成员和非成员数据
        new_attack_data = torch.cat([attack_in_features, attack_out_features], dim=0)
        new_victim_data = torch.cat([victim_in_features, victim_out_features], dim=0)
        
        # 创建标签
        attack_labels = torch.cat([torch.ones(attack_in_features.size(0)),
                                   torch.zeros(attack_out_features.size(0))], dim=0).unsqueeze(1)
        victim_labels = torch.cat([torch.ones(victim_in_features.size(0)),
                                   torch.zeros(victim_out_features.size(0))], dim=0).unsqueeze(1)
        
        # 保存路径 - 包含特征组合和参考模型配置 (model_num和query_num)
        # 从原始参考模型列表推断model_num
        if len(self.rapid_shadow_model_list) > 0 and isinstance(self.rapid_shadow_model_list[0], list):
            actual_model_num = len(self.rapid_shadow_model_list[0])
        else:
            actual_model_num = self.rapid_attack_in_confidences_raw.size(1) if self.rapid_attack_in_confidences_raw.dim() > 1 else 0
        
        save_folder = f"results/{self.dataset_name}_{self.model_name}/RAPID/rapid_attack_{self.feature_group}_r{actual_model_num}_q{self.query_num}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        attack_model_save_folder = save_folder
        
        attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        
        attack_train_dataloader = DataLoader(
            attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)
        
        # 创建攻击模型,输入维度根据特征数量自动确定
        # 使用指定的攻击模型架构 (mia_fc / mia_enhanced / mia_attention / mia_transformer)
        attack_model = BaseModel(
            self.attack_model_type, device=self.device, save_folder=attack_model_save_folder, 
            num_cls=1, input_dim=new_attack_data.size(1),
            optimizer=self.optimizer, lr=self.lr, weight_decay=self.weight_decay, 
            scheduler=self.scheduler, epochs=self.epochs)

        best_acc = 0
        best_tpr = 0
        print(f"使用攻击模型: {self.attack_model_type}, 特征组合: {self.feature_group}, 特征维度: {new_attack_data.size(1)}")
        print(f"结果保存至: {save_folder}")
        for epoch in range(self.epochs):
            train_acc, train_loss = attack_model.attack_train(attack_train_dataloader)
            test_acc, test_loss = attack_model.attack_test(attack_test_dataloader)
            test_acc_plot, test_loss_plot, ROC_label, ROC_confidence_score = attack_model.plot_test(attack_test_dataloader)
            ROC_confidence_score = np.nan_to_num(ROC_confidence_score,nan=np.nanmean(ROC_confidence_score))
            fpr, tpr, thresholds = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
            low = tpr[np.where(fpr<.001)[0][-1]]
            if low > best_tpr:
                best_tpr = low
                best_acc = test_acc
                attack_model.save(epoch, test_acc, test_loss)
                np.savez(f'{save_folder}/Roc_confidence_score.npz',acc=best_acc.cpu(),ROC_label=ROC_label,ROC_confidence_score=ROC_confidence_score)

            best_auc = roc_plot(ROC_label, ROC_confidence_score, plot=False)
        return best_tpr, best_auc, best_acc
        
    