#!/usr/bin/env python3
# UNet 模型测试脚本

import os

# 导入模块化组件
from test_config import parse_args
from tester import UNetTester

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    args = parse_args()
    tester = UNetTester(args)
    results, avg_iou, avg_dice, avg_ssim, avg_ms_ssim, valid_samples = tester.test()
    tester.print_results(results, avg_iou, avg_dice, avg_ssim, avg_ms_ssim, valid_samples)
    tester.save_results(results, avg_iou, avg_dice, avg_ssim, avg_ms_ssim, valid_samples)

if __name__ == "__main__":
    main()
