import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils import get_model, get_optimizer, get_scheduler, weight_init
import torch.distributed as dist
class BaseModel:
    def __init__(self, model_type, device="cuda", save_folder="", num_cls=10, input_dim=100, num_sub=2,
                 optimizer="", lr=0, weight_decay=0, scheduler="", epochs=0, attack_model_type=""):
        self.model = get_model(model_type, num_cls, input_dim, num_submodule=num_sub)
        self.model.to(device)
        self.model.apply(weight_init)
        self.device = device
        if epochs == 0:
            self.optimizer = None
            self.scheduler = None
        else:
            self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr, weight_decay)
            self.scheduler = get_scheduler(scheduler, self.optimizer, epochs)
        if model_type in ["mia_fc", "mia_fc_bn", "mia_enhanced", "mia_attention", "mia_transformer"]:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.save_pref = save_folder
        self.num_cls = num_cls

        if attack_model_type:
            self.attack_model = get_model(attack_model_type, num_cls*2, 2)
            self.attack_model.to(device)
            self.attack_model.apply(weight_init)
            self.attack_model_optim = get_optimizer("adam", self.attack_model.parameters(), lr=0.001, weight_decay=5e-4)

    def train(self, train_loader, log_pref="", distributed=False):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        if distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)

            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
            total_loss = total_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def attack_train(self, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            predicted = torch.round(torch.sigmoid(outputs))
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def attack_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}%, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def plot_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            ROC_confidence_score = torch.empty(0).to(self.device)
            ROC_label = torch.empty(0).to(self.device)
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
                total += targets.size(0)

                ROC_confidence_score = torch.cat((ROC_confidence_score, torch.sigmoid(outputs)))
                ROC_label = torch.cat((ROC_label, targets))

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss, torch.squeeze(ROC_label).cpu().numpy(), torch.squeeze(ROC_confidence_score).cpu().numpy()

    def save(self, epoch, acc, loss):
        save_path = f"{self.save_pref}/best.pth"
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'loss': loss,
            'state': self.model.state_dict()
        }
        torch.save(state, save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        acc = state['acc']
        if verbose:
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        self.model.load_state_dict(state['state'])
        return acc
    
    def predict_target_loss(self, data_loader):
        self.model.eval()

        loss_list = []
        target_list = []
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            # 添加数值稳定性：限制概率最小值，避免log(0)=-inf
            predicts = torch.clamp(predicts, min=1e-7, max=1.0)
            log_predicts = torch.log(predicts)
            losses = F.nll_loss(log_predicts, targets, reduction='none')
            losses = torch.unsqueeze(losses, 1)
            loss_list.append(losses.detach().data.cpu())
            target_list.append(targets.detach().data.cpu())

        losses = torch.cat(loss_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        return losses, targets
    
    def predict_with_statistics(self, data_loader):
        """
        提取模型预测的统计特征，用于 G1 策略
        返回：
            - losses: 目标类别的损失
            - max_conf: 最大置信度
            - entropy: 预测熵（不确定性）
            - margin: 前两类概率差
            - targets: 真实标签
        """
        self.model.eval()

        loss_list = []
        max_conf_list = []
        entropy_list = []
        margin_list = []
        target_list = []
        
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            log_predicts = torch.log(predicts)
            
            # 目标损失
            losses = F.nll_loss(log_predicts, targets, reduction='none')
            losses = torch.unsqueeze(losses, 1)
            
            # 最大置信度
            max_conf, _ = torch.max(predicts, dim=-1, keepdim=True)
            
            # 熵 (不确定性): -sum(p * log(p))
            entropy = -torch.sum(predicts * log_predicts, dim=-1, keepdim=True)
            
            # Margin: 前两类概率差
            top2_probs, _ = torch.topk(predicts, k=2, dim=-1)
            margin = top2_probs[:, 0:1] - top2_probs[:, 1:2]
            
            loss_list.append(losses.detach().data.cpu())
            max_conf_list.append(max_conf.detach().data.cpu())
            entropy_list.append(entropy.detach().data.cpu())
            margin_list.append(margin.detach().data.cpu())
            target_list.append(targets.detach().data.cpu())

        losses = torch.cat(loss_list, dim=0)
        max_confs = torch.cat(max_conf_list, dim=0)
        entropies = torch.cat(entropy_list, dim=0)
        margins = torch.cat(margin_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        
        return losses, max_confs, entropies, margins, targets
    