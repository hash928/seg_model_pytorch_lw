#!/usr/bin/env python3
"""
快速验证SAM1和SAM2模型的简单脚本
"""

import torch
from segment_anything import sam_model_registry
import sys
import os

# 添加SAM2模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
try:
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("警告: SAM2模块不可用，将跳过SAM2验证")

def quick_verify_sam1(checkpoint_path, model_type="vit_b"):
    """快速验证SAM1模型"""
    print(f"快速验证SAM1模型: {checkpoint_path}")
    print(f"模型类型: {model_type}")
    print("-" * 50)
    
    try:
        # 加载模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # 检查基本组件
        has_image_encoder = hasattr(sam, 'image_encoder')
        has_prompt_encoder = hasattr(sam, 'prompt_encoder')
        has_mask_decoder = hasattr(sam, 'mask_decoder')
        
        print(f"图像编码器: {'✅' if has_image_encoder else '❌'}")
        print(f"提示编码器: {'✅' if has_prompt_encoder else '❌'}")
        print(f"掩码解码器: {'✅' if has_mask_decoder else '❌'}")
        
        # 检查参数量
        total_params = sum(p.numel() for p in sam.parameters())
        print(f"总参数量: {total_params:,}")
        
        # 验证参数量范围
        if model_type == "vit_b" and 90_000_000 < total_params < 100_000_000:
            print("✅ 参数量符合SAM1 ViT-B模型")
        elif model_type == "vit_l" and 300_000_000 < total_params < 400_000_000:
            print("✅ 参数量符合SAM1 ViT-L模型")
        elif model_type == "vit_h" and 600_000_000 < total_params < 700_000_000:
            print("✅ 参数量符合SAM1 ViT-H模型")
        else:
            print("⚠️  参数量与预期不符")
        
        # 检查模型类名
        model_class = sam.__class__.__name__
        print(f"模型类名: {model_class}")
        
        if "Sam" in model_class:
            print("✅ 这是SAM模型")
        else:
            print("❌ 这可能不是SAM模型")
        
        print("\n🎉 SAM1验证完成！这看起来是一个有效的SAM1模型。")
        return True
        
    except Exception as e:
        print(f"❌ SAM1验证失败: {str(e)}")
        return False

def quick_verify_sam2(checkpoint_path, config_file="sam2_hiera_b+.yaml"):
    """快速验证SAM2模型"""
    print(f"快速验证SAM2模型: {checkpoint_path}")
    print(f"配置文件: {config_file}")
    print("-" * 50)
    
    if not SAM2_AVAILABLE:
        print("❌ SAM2模块不可用，无法验证SAM2模型")
        return False
    
    try:
        # 加载模型
        sam2 = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device="cpu",  # 使用CPU进行验证
            mode="eval"
        )
        
        # 检查基本组件
        has_image_encoder = hasattr(sam2, 'image_encoder')
        has_memory_attention = hasattr(sam2, 'memory_attention')
        has_memory_encoder = hasattr(sam2, 'memory_encoder')
        has_sam_mask_decoder = hasattr(sam2, 'sam_mask_decoder')
        
        print(f"图像编码器: {'✅' if has_image_encoder else '❌'}")
        print(f"记忆注意力: {'✅' if has_memory_attention else '❌'}")
        print(f"记忆编码器: {'✅' if has_memory_encoder else '❌'}")
        print(f"SAM掩码解码器: {'✅' if has_sam_mask_decoder else '❌'}")
        
        # 检查参数量
        total_params = sum(p.numel() for p in sam2.parameters())
        print(f"总参数量: {total_params:,}")
        
        # 验证参数量范围（SAM2的参数量通常比SAM1大）
        if 100_000_000 < total_params < 1_000_000_000:
            print("✅ 参数量符合SAM2模型范围")
        else:
            print("⚠️  参数量与预期不符")
        
        # 检查模型类名
        model_class = sam2.__class__.__name__
        print(f"模型类名: {model_class}")
        
        if "SAM2" in model_class or "Sam2" in model_class:
            print("✅ 这是SAM2模型")
        else:
            print("⚠️  这可能不是SAM2模型")
        
        # 检查配置相关的属性
        if hasattr(sam2, 'image_size'):
            print(f"图像尺寸: {sam2.image_size}")
        if hasattr(sam2, 'num_maskmem'):
            print(f"掩码记忆数量: {sam2.num_maskmem}")
        
        print("\n🎉 SAM2验证完成！这看起来是一个有效的SAM2模型。")
        return True
        
    except Exception as e:
        print(f"❌ SAM2验证失败: {str(e)}")
        return False

def quick_verify_sam(checkpoint_path, model_type="sam1", sam1_type="vit_b", sam2_config="sam2_hiera_b+.yaml"):
    """统一验证SAM1或SAM2模型"""
    print(f"🔍 开始验证SAM模型")
    print(f"模型类型: {model_type}")
    print(f"检查点路径: {checkpoint_path}")
    print("=" * 60)
    
    if model_type.lower() == "sam1":
        return quick_verify_sam1(checkpoint_path, sam1_type)
    elif model_type.lower() == "sam2":
        return quick_verify_sam2(checkpoint_path, sam2_config)
    else:
        print(f"❌ 不支持的模型类型: {model_type}")
        print("支持的模型类型: sam1, sam2")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python quick_verify_sam1.py <model_type> <checkpoint_path> [options]")
        print("")
        print("模型类型:")
        print("  sam1 <checkpoint_path> [sam1_type]     - 验证SAM1模型")
        print("  sam2 <checkpoint_path> [config_file]   - 验证SAM2模型")
        print("  auto <checkpoint_path>                 - 自动检测模型类型")
        print("")
        print("示例:")
        print("  python quick_verify_sam1.py sam1 sam_vit_b_01ec64.pth vit_b")
        print("  python quick_verify_sam1.py sam2 sam2_hiera_b+.pth sam2_hiera_b+.yaml")
        print("  python quick_verify_sam1.py auto sam_vit_b_01ec64.pth")
        sys.exit(1)
    
    # 检查是否是自动检测模式或旧格式
    if len(sys.argv) == 2:
        # 只有两个参数，可能是旧格式或自动检测
        checkpoint_path = sys.argv[1]
        print("🔍 自动检测模式")
        print(f"检查点路径: {checkpoint_path}")
        print("=" * 60)
        
        # 尝试自动检测模型类型
        success = False
        
        # 首先尝试SAM1
        print("尝试检测为SAM1模型...")
        for sam1_type in ["vit_b", "vit_l", "vit_h"]:
            try:
                if quick_verify_sam1(checkpoint_path, sam1_type):
                    print(f"✅ 成功检测为SAM1 {sam1_type}模型")
                    success = True
                    break
            except:
                continue
        
        # 如果SAM1失败，尝试SAM2
        if not success and SAM2_AVAILABLE:
            print("\n尝试检测为SAM2模型...")
            for config in ["sam2_hiera_b+.yaml", "sam2_hiera_l.yaml", "sam2_hiera_s.yaml", "sam2_hiera_t.yaml"]:
                try:
                    if quick_verify_sam2(checkpoint_path, config):
                        print(f"✅ 成功检测为SAM2模型 (配置: {config})")
                        success = True
                        break
                except:
                    continue
        
        if not success:
            print("❌ 无法自动检测模型类型")
            print("请手动指定模型类型:")
            print("  python quick_verify_sam1.py sam1 <checkpoint_path> [sam1_type]")
            print("  python quick_verify_sam1.py sam2 <checkpoint_path> [config_file]")
            sys.exit(1)
    
    elif len(sys.argv) >= 3:
        model_type = sys.argv[1]
        checkpoint_path = sys.argv[2]
        
        if model_type.lower() == "sam1":
            sam1_type = sys.argv[3] if len(sys.argv) > 3 else "vit_b"
            success = quick_verify_sam1(checkpoint_path, sam1_type)
        elif model_type.lower() == "sam2":
            sam2_config = sys.argv[3] if len(sys.argv) > 3 else "sam2_hiera_b+.yaml"
            success = quick_verify_sam2(checkpoint_path, sam2_config)
        elif model_type.lower() == "auto":
            # 自动检测模式
            print("🔍 自动检测模式")
            print(f"检查点路径: {checkpoint_path}")
            print("=" * 60)
            
            success = False
            
            # 首先尝试SAM1
            print("尝试检测为SAM1模型...")
            for sam1_type in ["vit_b", "vit_l", "vit_h"]:
                try:
                    if quick_verify_sam1(checkpoint_path, sam1_type):
                        print(f"✅ 成功检测为SAM1 {sam1_type}模型")
                        success = True
                        break
                except:
                    continue
            
            # 如果SAM1失败，尝试SAM2
            if not success and SAM2_AVAILABLE:
                print("\n尝试检测为SAM2模型...")
                for config in ["sam2_hiera_b+.yaml", "sam2_hiera_l.yaml", "sam2_hiera_s.yaml", "sam2_hiera_t.yaml"]:
                    try:
                        if quick_verify_sam2(checkpoint_path, config):
                            print(f"✅ 成功检测为SAM2模型 (配置: {config})")
                            success = True
                            break
                    except:
                        continue
            
            if not success:
                print("❌ 无法自动检测模型类型")
                print("请手动指定模型类型:")
                print("  python quick_verify_sam1.py sam1 <checkpoint_path> [sam1_type]")
                print("  python quick_verify_sam1.py sam2 <checkpoint_path> [config_file]")
                sys.exit(1)
        else:
            print(f"❌ 不支持的模型类型: {model_type}")
            print("支持的模型类型: sam1, sam2, auto")
            print("")
            print("用法:")
            print("  python quick_verify_sam1.py sam1 <checkpoint_path> [sam1_type]")
            print("  python quick_verify_sam1.py sam2 <checkpoint_path> [config_file]")
            print("  python quick_verify_sam1.py auto <checkpoint_path>")
            sys.exit(1)
    
    sys.exit(0 if success else 1) 