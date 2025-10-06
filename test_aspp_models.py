#!/usr/bin/env python3
# 测试ASPP模块是否正常工作

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import create_model, print_model_structure

def test_aspp_model(model_type, use_aspp=True, input_size=352):
    """测试带ASPP的模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_type} {'+ ASPP' if use_aspp else '(标准)'}")
    print(f"{'='*60}")
    
    try:
        # 创建模型
        model = create_model(model_type, 1, pretrained=False, use_aspp=use_aspp)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")
        
        # 创建测试输入
        test_input = torch.randn(1, 3, input_size, input_size).to(device)
        print(f"输入形状: {test_input.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"✓ {model_type} {'+ ASPP' if use_aspp else '(标准)'} 测试成功")
        
        return True
        
    except Exception as e:
        print(f"✗ {model_type} {'+ ASPP' if use_aspp else '(标准)'} 测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试ASPP模块...")
    
    # 测试的模型列表
    test_models = [
        'unet_base',
        'unet_resnet18', 
        'unet_resnet50',
        'unet_resnet101'
    ]
    
    success_count = 0
    total_count = 0
    
    for model_type in test_models:
        # 测试标准模型
        if test_aspp_model(model_type, use_aspp=False):
            success_count += 1
        total_count += 1
        
        # 测试ASPP模型
        if test_aspp_model(model_type, use_aspp=True):
            success_count += 1
        total_count += 1
    
    print(f"\n{'='*60}")
    print(f"测试结果: {success_count}/{total_count} 个模型测试成功")
    print(f"{'='*60}")
    
    if success_count == total_count:
        print("🎉 所有模型测试通过！ASPP模块工作正常")
    else:
        print("⚠️  部分模型测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
