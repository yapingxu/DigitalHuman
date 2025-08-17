#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义EchoMimic管道包装器
确保使用用户上传的文件
"""

import os
import sys
import yaml
from pathlib import Path

class CustomEchoMimicPipeline:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.validate_inputs()
    
    def load_config(self, config_path):
        """加载配置文件"""
        if not config_path:
            # 查找最新的动态配置
            temp_dir = Path(__file__).parent / "temp_uploads"
            config_files = list(temp_dir.glob("dynamic_config_*.yaml"))
            if config_files:
                config_path = max(config_files, key=os.path.getctime)
                print(f"📋 使用配置: {config_path}")
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ 配置加载成功")
            return config
        else:
            print("❌ 配置文件不存在")
            return {}
    
    def validate_inputs(self):
        """验证输入文件"""
        required_files = {
            'reference_image': '参考图像',
            'audio_path': '音频文件',
            'pose_dir': 'Pose目录'
        }
        
        print("🔍 验证输入文件...")
        for key, description in required_files.items():
            file_path = self.config.get(key)
            if file_path and Path(file_path).exists():
                print(f"   ✅ {description}: {file_path}")
            else:
                print(f"   ❌ {description}: 文件不存在 ({file_path})")
    
    def run_inference(self):
        """运行推理"""
        print("\n🚀 开始推理...")
        print(f"   📸 参考图像: {self.config.get('reference_image')}")
        print(f"   🎵 音频文件: {self.config.get('audio_path')}")
        print(f"   🤸 Pose目录: {self.config.get('pose_dir')}")
        
        # 在这里调用实际的EchoMimic推理
        # 确保传递正确的参数
        
        # 示例调用原始推理脚本
        inference_cmd = [
            "python", "inference.py",
            "--config", str(self.find_latest_config()),
            "--reference_image", self.config.get('reference_image'),
            "--audio_path", self.config.get('audio_path'),
            "--pose_dir", self.config.get('pose_dir')
        ]
        
        print("💻 执行命令:")
        print("   " + " ".join(inference_cmd))
        
        # 这里应该调用实际的推理代码
        # import subprocess
        # result = subprocess.run(inference_cmd, capture_output=True, text=True)
        # return result
        
        return "推理完成 (模拟)"
    
    def find_latest_config(self):
        """查找最新的配置文件"""
        temp_dir = Path(__file__).parent / "temp_uploads"
        config_files = list(temp_dir.glob("dynamic_config_*.yaml"))
        if config_files:
            return max(config_files, key=os.path.getctime)
        return None

if __name__ == "__main__":
    print("🎯 自定义EchoMimic管道包装器")
    print("=" * 50)
    
    pipeline = CustomEchoMimicPipeline()
    result = pipeline.run_inference()
    print(f"\n📊 结果: {result}")
