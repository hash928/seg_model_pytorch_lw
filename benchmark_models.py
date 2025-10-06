#!/usr/bin/env python3
# 模型性能基准测试脚本

import time
import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import create_model

def benchmark_model(model_type, input_size=352, batch_size=4, num_iterations=10):
    """测试模型性能"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_type}")
    print(f"{'='*60}")
    
    try:
        # 创建模型
        model = create_model(model_type, 1, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")
        
        # 创建测试输入
        test_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
        print(f"输入形状: {test_input.shape}")
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_input)
        
        # 测试前向传播时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # 计算性能指标
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = batch_size / avg_time
        
        print(f"总时间: {total_time:.4f}s")
        print(f"平均时间: {avg_time:.4f}s")
        print(f"FPS: {fps:.2f}")
        print(f"输出形状: {output.shape}")
        
        # 内存使用
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"GPU内存使用: {memory_used:.2f} MB")
        
        return {
            'model_type': model_type,
            'params': total_params,
            'avg_time': avg_time,
            'fps': fps,
            'memory_mb': memory_used if torch.cuda.is_available() else 0
        }
        
    except Exception as e:
        print(f"✗ {model_type} 测试失败: {str(e)}")
        return None

def main():
    """主测试函数"""
    print("开始模型性能基准测试...")
    
    # 测试的模型列表
    test_models = [
        'unet_base',
        'unet_resnet18', 
        'unet_resnet50',
        'fcn8s',
        'deeplabv3p_resnet50',
        'deeplabv3p_xception'
    ]
    
    results = []
    
    for model_type in test_models:
        result = benchmark_model(model_type, input_size=352, batch_size=2, num_iterations=5)
        if result:
            results.append(result)
    
    # 打印结果总结
    print(f"\n{'='*80}")
    print("性能测试结果总结")
    print(f"{'='*80}")
    print(f"{'模型':<20} {'参数量(M)':<12} {'时间(s)':<10} {'FPS':<8} {'内存(MB)':<10}")
    print(f"{'-'*80}")
    
    for result in results:
        params_m = result['params'] / 1e6
        print(f"{result['model_type']:<20} {params_m:<12.1f} {result['avg_time']:<10.4f} {result['fps']:<8.2f} {result['memory_mb']:<10.1f}")
    
    # 找出最快的模型
    if results:
        fastest = min(results, key=lambda x: x['avg_time'])
        print(f"\n🏆 最快模型: {fastest['model_type']} ({fastest['avg_time']:.4f}s)")
        
        # 找出参数量最少的模型
        smallest = min(results, key=lambda x: x['params'])
        print(f"📦 最轻量模型: {smallest['model_type']} ({smallest['params']/1e6:.1f}M参数)")
        
        # 找出FPS最高的模型
        highest_fps = max(results, key=lambda x: x['fps'])
        print(f"⚡ 最高FPS: {highest_fps['model_type']} ({highest_fps['fps']:.2f} FPS)")

if __name__ == "__main__":
    main()
