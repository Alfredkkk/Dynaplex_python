import json
import sys
import torch
import os
import time
from datetime import datetime

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from dp import dynaplex
from dp.utils.early_stopping import EarlyStopping
from scripts.networks.gc_lsn_mlp import ActorMLP

# 快速测试版本 - 使用极小的参数
print("=== 快速测试版本 - 使用极小参数 ===")

# 设置训练参数
MAX_EPOCH = 20  # 减少轮次
num_gens = 2  # 只测试2代而不是5代

# 确定是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义MDP配置
vars = {
    "id": "lost_sales",
    "p": 4.0,        # 缺货成本参数
    "h": 1.0,        # 库存持有成本参数
    "leadtime": 3,   # 订单交付时间
    "discount_factor": 1.0,
    "demand_dist": {
        "type": "poisson",
        "mean": 4.0
    }
}

print("正在初始化MDP...")
start_time = time.time()
mdp = dynaplex.get_mdp(**vars)
print(f"MDP初始化完成，用时 {time.time() - start_time:.2f} 秒")

num_valid_actions = mdp.num_valid_actions()
num_features = mdp.num_flat_features()

print(f"特征数量: {num_features}, 有效动作数量: {num_valid_actions}")

# 使用基础策略
print("加载基础策略...")
base_policy = mdp.get_policy("base_stock")

# 配置样本生成器 - 使用极小的样本数量，以便快速测试
# N = 训练样本数量, M = 模拟步数
print("配置样本生成器...")
sample_generator = dynaplex.get_sample_generator(mdp, N=500, M=100)  # 极小样本量

save_filename = 'gc_lsn_test'

def policy_path(gen):
    return dynaplex.filepath(mdp.identifier(), f'{save_filename}_{gen}')

def sample_path(gen):
    return dynaplex.filepath(mdp.identifier(), f'samples_test_{gen}.json')

# 训练循环
print("=" * 80)
print(f"开始训练 {num_gens} 代策略")
print("=" * 80)

total_start_time = time.time()

for gen in range(0, num_gens):
    gen_start_time = time.time()
    print(f"\n开始第 {gen+1}/{num_gens} 代训练")
    print("-" * 60)
    
    if gen > 0:
        print(f"加载第 {gen} 代策略...")
        policy = dynaplex.load_policy(mdp, policy_path(gen))
    else:
        print("使用基础策略...")
        policy = base_policy

    save_model_path = policy_path(gen + 1)
    
    # 生成样本并显示进度
    print(f"生成样本...")
    sample_start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    sample_generator.generate_samples(policy, sample_path(gen))
    print(f"结束时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"样本生成完成，用时 {time.time() - sample_start_time:.2f} 秒")

    print(f"处理样本数据...")
    process_start_time = time.time()
    with open(sample_path(gen), 'r') as json_file:
        sample_data = json.load(json_file)['samples']
        print(f"样本数量: {len(sample_data)}")

        tensor_y = torch.LongTensor([sample['action_label'] for sample in sample_data])
        tensor_mask = torch.BoolTensor([sample['allowed_actions'] for sample in sample_data])
        tensor_x = torch.FloatTensor([sample['features'] for sample in sample_data])

        min_val = torch.finfo(tensor_x.dtype).min

        # 创建模型
        print(f"创建模型...")
        model = ActorMLP(input_dim=mdp.num_flat_features(), 
                         hidden_dim=[64, 64],  # 更小的网络
                         output_dim=num_valid_actions,
                         min_val=torch.finfo(torch.float).min)

        # 移动模型到设备
        model.to(device)  

        # 创建数据集
        dataset = TensorDataset(tensor_x, tensor_y, tensor_mask)

        # 分割训练集和验证集
        train_dim = int(0.95 * len(dataset))
        valid_dim = len(dataset) - train_dim

        valid_dataset = torch.utils.data.Subset(dataset, range(train_dim, len(dataset)))
        train_dataset = torch.utils.data.Subset(dataset, range(train_dim))
        
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")

        # 数据加载器 - 使用较大的batch size以减少训练时间
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        valid_dataloader = DataLoader(valid_dataset, batch_size=32, num_workers=0)
        
        print(f"数据处理完成，用时 {time.time() - process_start_time:.2f} 秒")

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

        # 损失函数
        loss_function = nn.NLLLoss()
        log_softmax = nn.LogSoftmax(dim=-1)

        # 训练跟踪
        avg_train_losses = []
        avg_valid_losses = []

        # 早停机制 - 减少早停参数
        early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.001)

        print(f"开始模型训练...")
        train_start_time = time.time()
        
        # 训练循环
        for ep in range(MAX_EPOCH):
            epoch_start = time.time()
            train_losses = []
            valid_losses = []

            # 训练模型
            model.train()
            for i, (inputs, targets, data_mask) in enumerate(dataloader):
                # 进度指示
                if i % 5 == 0 or i == len(dataloader) - 1:
                    progress = (i+1) / len(dataloader) * 100
                    print(f"\r训练批次 {i+1}/{len(dataloader)} ({progress:.1f}%)", end="", flush=True)
                
                # 移动数据到设备
                inputs = inputs.to(device)
                targets = targets.to(device)
                data_mask = data_mask.to(device)

                optimizer.zero_grad()
                outputs = model.training_forward(inputs, data_mask)
                log_outputs = log_softmax(outputs)

                loss = loss_function(log_outputs, targets)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
            
            print()  # 换行
            
            # 验证模型
            model.eval()
            with torch.no_grad():
                for inputs, targets, data_mask in valid_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    data_mask = data_mask.to(device)

                    outputs = model.training_forward(inputs, data_mask)
                    log_outputs = log_softmax(outputs)

                    loss = loss_function(log_outputs, targets)
                    valid_losses.append(loss.item())

            # 计算平均损失
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_time = time.time() - epoch_start
            print(f'轮次 {ep + 1}/{MAX_EPOCH} - 训练损失: {train_loss:.5f} - 验证损失: {valid_loss:.5f} - 耗时: {epoch_time:.2f}秒')

            # 检查早停
            save_model, early_stop = early_stopping(valid_loss, model)

            if save_model:
                json_info = {
                    'id': 'NN_Policy', 
                    'gen': gen + 1,
                    'num_inputs': num_features, 
                    'num_outputs': num_valid_actions,
                    'input_type': 'tensor',
                    'nn_architecture': {
                        'type': 'mlp',
                        'hidden_layers': [64, 64]
                    }
                }

                save_start = time.time()
                dynaplex.save_policy(model, json_info, save_model_path, device)
                print(f"模型保存到 {save_model_path}，耗时 {time.time() - save_start:.2f}秒")

            if early_stop:
                print("早停触发，训练结束")
                break
        
        print(f"第 {gen+1} 代训练完成，总用时 {time.time() - train_start_time:.2f}秒")

    print(f"第 {gen+1} 代完成，总用时 {time.time() - gen_start_time:.2f}秒")

# 加载和比较所有策略
print("\n" + "=" * 80)
print("训练完成，比较所有策略性能")
print("=" * 80)

policies = [base_policy]
policy_names = ["Base"]

for i in range(1, num_gens + 1):
    try:
        print(f"加载第 {i} 代策略...")
        load_path = policy_path(i)
        policy = dynaplex.load_policy(mdp, load_path)
        policies.append(policy)
        policy_names.append(f"Gen {i}")
    except Exception as e:
        print(f"无法加载第 {i} 代策略: {e}")

print("进行策略评估...")
# 使用更少的trajectory来加速评估
comparer = dynaplex.get_comparer(mdp, number_of_trajectories=20, periods_per_trajectory=20)
comparison = comparer.compare(policies)

print("\n策略比较结果:")
print("-" * 60)
print(f"{'策略':<10} {'平均回报':<15}")
print("-" * 60)
for i, item in enumerate(comparison):
    if isinstance(item, dict):
        # 检查字典中包含哪些键
        if 'mean' in item:
            print(f"{policy_names[i]:<10} {item['mean']:<15.2f}")
        else:
            # 如果没有'mean'键，直接打印整个项
            print(f"{policy_names[i]:<10} {item}")
    else:
        # 如果不是字典，直接打印值
        print(f"{policy_names[i]:<10} {item:<15.2f}")

# 保存最终策略
print("\n保存最终策略...")
try:
    final_policy = dynaplex.load_policy(mdp, policy_path(num_gens))
    final_save_path = dynaplex.filepath("", "GC-LSN_test")
    dynaplex.save_policy(final_policy, 
                         {
                             'id': 'NN_Policy', 
                             'gen': num_gens,
                             'num_inputs': num_features, 
                             'num_outputs': num_valid_actions,
                             'input_type': 'tensor',
                             'nn_architecture': {
                                 'type': 'mlp',
                                 'hidden_layers': [64, 64]
                             }
                         }, 
                         final_save_path, 
                         device)
    print(f"最终策略保存为 GC-LSN_test.pth 和 GC-LSN_test.json")
except Exception as e:
    print(f"保存最终策略时出错: {e}")

# 训练总结
total_time = time.time() - total_start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("\n" + "=" * 80)
print(f"训练总结")
print("=" * 80)
print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
print(f"训练代数: {num_gens}")
print(f"最终模型保存位置: {final_save_path}")
print("=" * 80) 