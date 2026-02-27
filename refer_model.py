import argparse
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from models.base_model import BaseModel
from datasets import get_dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import seed_worker, get_optimizer, get_scheduler, print_set, init_distributed, cleanup


# 设置命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('device', default=0, type=int, help="要使用的GPU ID")
parser.add_argument('config_path', default=0, type=str, help="配置文件路径")
# parser.add_argument('--device', default=0, type=int, help="要使用的GPU ID")
parser.add_argument('--distributed', default=False, type=bool, help="是否启用分布式训练")
parser.add_argument('--world_size', default=1, type=int, help="分布式训练的进程数")
parser.add_argument('--dataset_name', default='mnist', type=str, help="数据集名称")
parser.add_argument('--model_name', default='mnist', type=str, help="模型名称")
parser.add_argument('--num_cls', default=10, type=int, help="分类任务的类别数")
parser.add_argument('--input_dim', default=3, type=int, help="输入数据的通道数")
parser.add_argument('--seed', default=42, type=int, help="随机种子，用于保证实验可复现")
parser.add_argument('--model_num', default='2', type=int, help="要训练的参考模型数量")
parser.add_argument('--batch_size', default=128, type=int, help="训练时的批处理大小")
parser.add_argument('--epochs', default=100, type=int, help="训练的总轮数")
parser.add_argument('--early_stop', default=0, type=int, help="早停的耐心值，0表示不使用早停")
parser.add_argument('--lr', default=0.1, type=float, help="学习率")
parser.add_argument('--weight_decay', default=5e-4, type=float, help="权重衰减系数")
parser.add_argument('--optimizer', default="sgd", type=str, help="优化器类型 (例如: sgd, adam)")
parser.add_argument('--scheduler', default="cosine", type=str, help="学习率调度器类型 (例如: cosine, step)")

def main_worker(rank, world_size, args):
    """主工作函数，用于训练参考模型，支持分布式训练"""
    # 初始化分布式环境
    init_distributed(rank, world_size)
    print(f"在 rank {rank} 上运行, 总进程数: {world_size}")

    # 设置随机种子以确保结果可复现
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 根据是否为分布式训练来确定设备
    if args.distributed == False:
        device = f"cuda:{args.device}"
    else:
        device = f"cuda:{rank}"

    # 设置CUDNN以保证确定性，这对于可复现性很重要
    cudnn.deterministic = True
    cudnn.benchmark = False

    # 定义结果保存的基础路径
    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    if rank == 0:
        print(f"结果保存基础路径: {base_folder}")

    # 加载数据集
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    aug_trainset = get_dataset(args.dataset_name, train=True, augment=True) # 带数据增强的训练集
    aug_testset = get_dataset(args.dataset_name, train=False, augment=True) # 带数据增强的测试集

    # 如果没有单独的测试集，则总数据集就是训练集；否则，合并训练集和测试集
    if testset is None:
        total_dataset = trainset
        aug_total_dataset = aug_trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
        aug_total_dataset = ConcatDataset([aug_trainset, aug_testset])
    total_size = len(total_dataset)

    # 从预存的文件中加载数据索引
    data_path = f"{base_folder}/data_index.pkl"
    if rank == 0:
        print(f"总数据量: {total_size}")

    with open(data_path, 'rb') as f:
        # 这个文件包含了多个数据子集的索引，这里只关心用于训练参考模型的部分
        _, _, _, _, tuning_train_list, tuning_test_list = pickle.load(f)

    # 合并这两部分数据索引，形成一个用于训练参考模型的数据池
    combined_list = tuning_train_list + tuning_test_list
    if rank == 0: 
        print(f"用于训练参考模型的训练数据量: {len(tuning_train_list)}, "
              f"测试数据量: {len(tuning_test_list)}")

        # --- 这部分代码在原脚本中用于加载受害者和影子模型，但在当前逻辑下不会被执行 ---
        # 因为后续的循环会重新划分数据，所以这里加载的模型实际上没有被使用
        # victim_model_save_folder = f"{base_folder}/victim_model"
        # if not os.path.exists(f"{victim_model_save_folder}/best.pth"):
        #     raise FileNotFoundError("找不到预训练的受害者模型")
        # victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        # victim_model.load(f"{victim_model_save_folder}/best.pth")
        # shadow_model_save_folder = f"{base_folder}/shadow_model"
        # if not os.path.exists(f"{shadow_model_save_folder}/best.pth"):
        #     raise FileNotFoundError("找不到预训练的影子模型")
        # shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        # shadow_model.load(f"{shadow_model_save_folder}/best.pth")

    # --- 开始训练参考模型 ---
    print(f"开始训练参考模型...")
    # 循环 `model_num` 次，训练多个参考模型
    for idx in range(args.model_num):
        # 1. 数据准备：为每个参考模型创建独特的训练集和测试集
        random.shuffle(combined_list) # 随机打乱数据池的索引
        split_index = len(combined_list) // 2 # 计算对半分割点
        tuning_train_list = combined_list[:split_index] # 前一半作为训练集
        tuning_test_list = combined_list[split_index:2*split_index] # 后一半作为测试集

        # 使用Subset从增强数据集中创建训练和测试数据子集
        victim_train_dataset = Subset(aug_total_dataset, tuning_train_list)
        victim_test_dataset = Subset(aug_total_dataset, tuning_test_list)
        
        # 2. 创建数据加载器 (DataLoader)
        if args.distributed:
            # 分布式训练需要使用 DistributedSampler
            train_sampler = DistributedSampler(victim_train_dataset, num_replicas=world_size, rank=rank)
            victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker)
        else:
            # 单机训练则直接设置 shuffle=True
            victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, worker_init_fn=seed_worker)

        # 只有主进程 (rank 0) 需要测试加载器
        if rank == 0:
            victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)

        
        # 3. 初始化模型、优化器和调度器
        tuning_victim_model_save_folder = f"{base_folder}/RAPID/reference_model_{idx}"
        if not os.path.exists(tuning_victim_model_save_folder):
            os.makedirs(tuning_victim_model_save_folder, exist_ok=True)

        # 创建一个新的基础模型实例用于本次训练
        trans_victim_model = BaseModel(
                args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, 
                save_folder=tuning_victim_model_save_folder, device=device, 
                optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, 
                scheduler=args.scheduler, epochs=args.epochs)

        # 为模型配置优化器和学习率调度器
        if args.distributed:
            # 在分布式模式下，需要将模型包装在 DDP 中
            trans_victim_model_model = DDP(trans_victim_model.model, device_ids=[rank])
            trans_victim_model.optimizer = get_optimizer(args.optimizer,
                                                filter(lambda p: p.requires_grad, trans_victim_model_model.parameters()), 
                                                lr=args.lr, weight_decay=args.weight_decay)
        else:
            trans_victim_model.optimizer = get_optimizer(args.optimizer, 
                                                filter(lambda p: p.requires_grad, trans_victim_model.model.parameters()), 
                                                lr=args.lr, weight_decay=args.weight_decay)
        trans_victim_model.scheduler = get_scheduler(args.scheduler, 
                                            trans_victim_model.optimizer, args.epochs)
        
        # 4. 执行训练和评估循环
        best_acc = 0 # 记录最佳测试准确率
        count = 0 # 用于早停的计数器
        for epoch in range(args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch) # 在每个epoch开始时设置sampler的epoch

            # 训练模型
            train_acc, train_loss = trans_victim_model.train(victim_train_loader, f"Epoch {epoch} Reference_{idx} Victim Train", args.distributed)

            # 只有主进程 (rank 0) 执行评估和模型保存
            if rank == 0:
                test_acc, test_loss = trans_victim_model.test(victim_test_loader, f"Epoch {epoch} Reference_{idx} Victim Test")

                # 如果当前测试准确率更高，则保存模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    trans_victim_model.save(epoch, test_acc, test_loss)
                    count = 0 # 重置早停计数器
                elif args.early_stop > 0:
                    # 如果启用了早停，并且性能没有提升，则增加计数
                    count += 1
                    if count > args.early_stop:
                        print(f"在 Epoch {epoch} 触发早停")
                        break
    # 清理分布式训练环境
    cleanup()

if __name__ == '__main__':
    # 解析命令行参数
    args = parser.parse_args()

    # 从JSON配置文件中加载参数，并允许命令行参数覆盖它们
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    # 打印最终使用的所有参数
    print_set(args)

    # 根据是否启用分布式训练来启动主工作函数
    if args.distributed:
        # 使用 mp.spawn 启动多个进程进行分布式训练
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size)
    else:
        # 单机模式下直接调用
        main_worker(0, 1, args)
