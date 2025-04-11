import json
import sys
import torch
import os
import time
import threading
from datetime import datetime

from dp import dynaplex

# 用于测试创建好的神经网络策略是否能正常使用
print("=== 神经网络策略加载和样本生成调试脚本 ===")

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

# 1. 测试内置策略生成样本
print("\n\n=== 测试1: 使用内置策略生成样本 ===")
print("加载基础策略...")
base_policy = mdp.get_policy("base_stock")

# 配置样本生成器 - 极小样本量用于测试
print("配置样本生成器...")
sample_generator = dynaplex.get_sample_generator(mdp, N=100, M=20)  # 极小样本量

print("使用base_stock策略生成样本...")
base_sample_path = dynaplex.filepath(mdp.identifier(), "test_base_samples.json")
sample_start_time = time.time()
sample_generator.generate_samples(base_policy, base_sample_path)
print(f"样本生成完成，用时 {time.time() - sample_start_time:.2f} 秒")

# 2. 尝试加载已有的GC-LSN策略
print("\n\n=== 测试2: 尝试加载已有的GC-LSN策略 ===")

# 设置操作完成标志
operation_done = False

def run_with_timeout(func, timeout=60):
    """使用线程来实现超时功能"""
    result = [None]
    error = [None]
    done_flag = [False]
    
    def worker():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e
        finally:
            done_flag[0] = True
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    
    start_time = time.time()
    thread.start()
    thread.join(timeout)
    
    if not done_flag[0]:
        return None, TimeoutError(f"操作超时 (超过{timeout}秒)")
    elif error[0] is not None:
        return None, error[0]
    else:
        return result[0], None

try:
    # 测试加载GC-LSN_test.pth，如果之前已生成
    test_policy_path = dynaplex.filepath("", "GC-LSN_test")
    
    print(f"尝试加载GC-LSN_test策略...")
    
    def load_policy():
        return dynaplex.load_policy(mdp, test_policy_path)
    
    gc_policy, error = run_with_timeout(load_policy, 30)
    
    if error:
        if isinstance(error, TimeoutError):
            print(f"警告: 加载策略超时: {error}")
        else:
            print(f"加载GC-LSN_test策略时出错: {error}")
        print("尝试加载其他可能存在的策略...")
        
        # 尝试加载第一次训练的策略
        gen1_policy_path = dynaplex.filepath(mdp.identifier(), "gc_lsn_test_1")
        
        def load_gen1_policy():
            return dynaplex.load_policy(mdp, gen1_policy_path)
        
        print(f"尝试加载gen1策略: {gen1_policy_path}")
        gc_policy, error = run_with_timeout(load_gen1_policy, 30)
        
        if error:
            if isinstance(error, TimeoutError):
                print(f"警告: 加载gen1策略超时: {error}")
            else:
                print(f"加载gen1策略时出错: {error}")
            gc_policy = None
    
    if gc_policy is not None:
        print("策略加载成功")
        
        print("使用加载的策略生成少量样本...")
        nn_sample_path = dynaplex.filepath(mdp.identifier(), "test_loaded_nn_samples.json")
        
        # 记录开始时间
        print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
        sample_start_time = time.time()
        
        # 使用极小的样本量
        def generate_samples():
            tiny_sample_generator = dynaplex.get_sample_generator(mdp, N=10, M=5)
            tiny_sample_generator.generate_samples(gc_policy, nn_sample_path)
        
        _, error = run_with_timeout(generate_samples, 60)
        
        if error:
            if isinstance(error, TimeoutError):
                print(f"警告: 样本生成超时: {error}")
            else:
                print(f"样本生成时出错: {error}")
        else:
            print(f"结束时间: {datetime.now().strftime('%H:%M:%S')}")
            print(f"样本生成完成，用时 {time.time() - sample_start_time:.2f} 秒")
    else:
        print("无法加载已有策略，尝试创建新策略")
        
except Exception as e:
    print(f"加载策略时出错: {e}")

# 3. 创建和保存一个全新的简单神经网络策略
print("\n\n=== 测试3: 创建和保存简单神经网络策略 ===")

try:
    # 直接创建一个简单的神经网络策略
    nn_config = {
        'id': 'Simple_NN_Policy', 
        'num_inputs': num_features, 
        'num_outputs': num_valid_actions,
        'input_type': 'tensor',
        'nn_architecture': {
            'type': 'mlp',
            'hidden_layers': [16, 16]  # 非常小的网络
        }
    }
    
    nn_save_path = dynaplex.filepath(mdp.identifier(), "simple_nn_policy")
    
    print(f"创建简单神经网络策略...")
    
    def create_mlp_policy():
        return dynaplex.create_mlp_policy(mdp, nn_config)
    
    nn_policy, error = run_with_timeout(create_mlp_policy, 30)
    
    if error:
        if isinstance(error, TimeoutError):
            print(f"警告: 创建策略超时: {error}")
        else:
            print(f"创建神经网络策略时出错: {error}")
        nn_policy = None
    else:
        print("神经网络策略创建成功")
        
        # 保存策略
        print(f"保存策略到 {nn_save_path}...")
        def save_policy():
            dynaplex.save_policy(nn_policy, nn_config, nn_save_path, device)
        
        _, error = run_with_timeout(save_policy, 30)
        
        if error:
            if isinstance(error, TimeoutError):
                print(f"警告: 保存策略超时: {error}")
            else:
                print(f"保存策略时出错: {error}")
        else:
            print("策略保存成功")
            
            # 使用新创建的策略生成样本
            print("\n使用新创建的策略生成样本...")
            new_sample_path = dynaplex.filepath(mdp.identifier(), "test_new_nn_samples.json")
            
            # 记录开始时间
            print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
            sample_start_time = time.time()
            
            def generate_samples():
                tiny_sample_generator = dynaplex.get_sample_generator(mdp, N=10, M=5)
                tiny_sample_generator.generate_samples(nn_policy, new_sample_path)
            
            _, error = run_with_timeout(generate_samples, 60)
            
            if error:
                if isinstance(error, TimeoutError):
                    print(f"警告: 样本生成超时: {error}")
                else:
                    print(f"样本生成时出错: {error}")
            else:
                print(f"结束时间: {datetime.now().strftime('%H:%M:%S')}")
                print(f"样本生成完成，用时 {time.time() - sample_start_time:.2f} 秒")
                
except Exception as e:
    print(f"创建或使用神经网络策略时出错: {e}")

print("\n测试完成!") 