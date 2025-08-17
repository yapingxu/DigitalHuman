#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复后的 EchoMimicV2 推理脚本
解决了组件加载和 audio_guider 的问题
"""
import os
import sys
import torch
import argparse
import yaml
from pathlib import Path
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.pose_encoder import PoseEncoder
from src.models.whisper.audio2feature import load_audio_model

def parse_args():
    parser = argparse.ArgumentParser(description="EchoMimicV2 推理脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dtype", type=str, default="fp16", help="数据类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_models_manually(config, device="cuda", dtype=torch.float16):
    """手动加载所有组件 - 修复版本"""
    print("🔧 开始加载模型组件...")
    
    # 设置模型路径
    if "BadToBest/EchoMimicV2" in str(config.get('model_path', '')):
        pretrained_model_path = "BadToBest/EchoMimicV2"
    else:
        pretrained_model_path = config.get('model_path', './pretrained_weights')
    
    components = {}
    
    try:
        # 1. 加载 VAE
        print("📦 加载 VAE...")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, 
            subfolder="vae",
            torch_dtype=dtype
        ).to(device)
        components['vae'] = vae
        print("✅ VAE 加载完成")
        
        # 2. 加载调度器
        print("📦 加载 Scheduler...")
        scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_path, 
            subfolder="scheduler"
        )
        components['scheduler'] = scheduler
        print("✅ Scheduler 加载完成")
        
        # 3. 加载 reference_unet
        print("📦 加载 Reference UNet...")
        reference_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path,
            subfolder="reference_unet",
            torch_dtype=dtype
        ).to(device)
        components['reference_unet'] = reference_unet
        print("✅ Reference UNet 加载完成")
        
        # 4. 加载 denoising_unet  
        print("📦 加载 Denoising UNet...")
        denoising_unet = EMOUNet3DConditionModel.from_pretrained(
            pretrained_model_path,
            subfolder="denoising_unet", 
            torch_dtype=dtype
        ).to(device)
        components['denoising_unet'] = denoising_unet
        print("✅ Denoising UNet 加载完成")
        
        # 5. 加载 pose_encoder
        print("📦 加载 Pose Encoder...")
        pose_encoder = PoseEncoder.from_pretrained(
            pretrained_model_path,
            subfolder="pose_encoder",
            torch_dtype=dtype
        ).to(device)
        components['pose_encoder'] = pose_encoder
        print("✅ Pose Encoder 加载完成")
        
        # 6. 加载 audio_guider (Audio2Feature) - 关键修复！
        print("📦 加载 Audio Processor...")
        
        # 尝试多个可能的音频模型路径
        audio_model_paths = [
            "./pretrained_weights_old/audio_processor/tiny.pt",
            "./pretrained_weights/audio_processor/tiny.pt", 
            config.get('audio_model_path', './pretrained_weights/audio_processor/tiny.pt')
        ]
        
        audio_guider = None
        for audio_path in audio_model_paths:
            if os.path.exists(audio_path):
                try:
                    audio_guider = load_audio_model(model_path=audio_path, device=device)
                    print(f"✅ Audio Processor 加载完成: {audio_path}")
                    break
                except Exception as e:
                    print(f"⚠️ 音频模型路径 {audio_path} 加载失败: {e}")
                    continue
        
        if audio_guider is None:
            raise FileNotFoundError(f"❌ 未找到可用的音频模型文件。尝试的路径: {audio_model_paths}")
        
        components['audio_guider'] = audio_guider
        
        print("🎉 所有组件加载完成！")
        return components
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    args = parse_args()
    
    print(f"🚀 启动 EchoMimicV2 推理...")
    print(f"📁 配置文件: {args.config}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备和数据类型
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        # 手动加载所有组件
        components = load_models_manually(config, device, dtype)
        
        # 创建管道
        print("🎭 创建 EchoMimicV2 管道...")
        pipeline = EchoMimicV2Pipeline(**components)
        
        # 获取推理参数
        face_img_path = config.get('reference_image')
        audio_path = config.get('audio_path')
        pose_dir = config.get('pose_dir')
        output_dir = config.get('output_dir', './outputs')
        
        print(f"🖼️  参考图像: {face_img_path}")
        print(f"🎵 音频文件: {audio_path}")
        print(f"🤖 姿态目录: {pose_dir}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行推理
        print("⚡ 开始生成数字人视频...")
        result = pipeline(
            face_img_path=face_img_path,
            pose_dir=pose_dir, 
            audio_path=audio_path,
            width=config.get('width', 512),
            height=config.get('height', 512),
            length=config.get('length', None),  # 自动根据音频长度确定
            guidance_scale=config.get('guidance_scale', 2.0),
            num_inference_steps=config.get('num_inference_steps', 25),
            generator=torch.Generator(device=device).manual_seed(args.seed)
        )
        
        print("✅ 推理完成！")
        print(f"📁 输出目录: {output_dir}")
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()