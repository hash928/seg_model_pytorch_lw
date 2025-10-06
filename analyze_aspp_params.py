#!/usr/bin/env python3
# 分析ASPP模块对UNet参数量的影响

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import create_model

def analyze_model_params(model, model_name):
    """分析模型参数分布"""
    print(f"\n{'='*60}")
    print(f"分析模型: {model_name}")
    print(f"{'='*60}")
    
    total_params = 0
    encoder_params = 0
    decoder_params = 0
    aspp_params = 0
    
    # 按模块统计参数
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if 'encoder' in name:
            encoder_params += param_count
        elif 'decoder' in name:
            decoder_params += param_count
        elif 'aspp' in name:
            aspp_params += param_count
    
    print(f"总参数量: {total_params:,}")
    print(f"Encoder参数量: {encoder_params:,}")
    print(f"Decoder参数量: {decoder_params:,}")
    if aspp_params > 0:
        print(f"ASPP参数量: {aspp_params:,}")
    
    return total_params, encoder_params, decoder_params, aspp_params

def analyze_decoder_structure(model, model_name):
    """分析decoder结构"""
    print(f"\n{model_name} Decoder结构分析:")
    print("-" * 40)
    
    decoder_params = 0
    ups_params = 0
    decodes_params = 0
    classifier_params = 0
    aspp_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        if 'decoder' in name:
            decoder_params += param_count
            
            if 'ups' in name:
                ups_params += param_count
            elif 'decodes' in name:
                decodes_params += param_count
            elif 'classifier' in name:
                classifier_params += param_count
            elif 'aspp' in name:
                aspp_params += param_count
    
    print(f"上采样层参数量: {ups_params:,}")
    print(f"解码层参数量: {decodes_params:,}")
    print(f"分类器参数量: {classifier_params:,}")
    if aspp_params > 0:
        print(f"ASPP模块参数量: {aspp_params:,}")
    
    return decoder_params, ups_params, decodes_params, classifier_params, aspp_params

def main():
    """主分析函数"""
    print("UNet + ASPP 参数量分析")
    print("=" * 60)
    
    # 创建模型
    unet_standard = create_model('unet_base', 1, pretrained=False, use_aspp=False)
    unet_aspp = create_model('unet_base', 1, pretrained=False, use_aspp=True)
    
    # 分析标准UNet
    std_total, std_encoder, std_decoder, std_aspp = analyze_model_params(unet_standard, "标准UNet")
    std_decoder, std_ups, std_decodes, std_classifier, std_aspp = analyze_decoder_structure(unet_standard, "标准UNet")
    
    # 分析ASPP UNet
    aspp_total, aspp_encoder, aspp_decoder, aspp_aspp = analyze_model_params(unet_aspp, "UNet + ASPP")
    aspp_decoder, aspp_ups, aspp_decodes, aspp_classifier, aspp_aspp = analyze_decoder_structure(unet_aspp, "UNet + ASPP")
    
    # 对比分析
    print(f"\n{'='*60}")
    print("对比分析")
    print(f"{'='*60}")
    
    print(f"总参数量变化: {aspp_total - std_total:,} ({aspp_total - std_total:+,})")
    print(f"Encoder参数量变化: {aspp_encoder - std_encoder:,} ({aspp_encoder - std_encoder:+,})")
    print(f"Decoder参数量变化: {aspp_decoder - std_decoder:,} ({aspp_decoder - std_decoder:+,})")
    
    print(f"\nDecoder内部变化:")
    print(f"上采样层变化: {aspp_ups - std_ups:,} ({aspp_ups - std_ups:+,})")
    print(f"解码层变化: {aspp_decodes - std_decodes:,} ({aspp_decodes - std_decodes:+,})")
    print(f"分类器变化: {aspp_classifier - std_classifier:,} ({aspp_classifier - std_classifier:+,})")
    print(f"ASPP模块: {aspp_aspp:,} (新增)")
    
    # 解释原因
    print(f"\n{'='*60}")
    print("参数量减少的原因分析")
    print(f"{'='*60}")
    
    print("1. ASPP模块替换了decoder的第一层上采样操作")
    print("2. 标准UNet的decoder第一层上采样参数量较大")
    print("3. ASPP模块虽然复杂，但参数量相对较少")
    print("4. 这种设计实际上优化了网络结构")
    
    # 计算具体的参数变化
    decoder_reduction = std_decoder - aspp_decoder
    aspp_addition = aspp_aspp
    net_change = aspp_addition - decoder_reduction
    
    print(f"\n具体计算:")
    print(f"Decoder减少的参数量: {decoder_reduction:,}")
    print(f"ASPP模块增加的参数量: {aspp_addition:,}")
    print(f"净变化: {net_change:,} ({net_change:+,})")

if __name__ == "__main__":
    main()

