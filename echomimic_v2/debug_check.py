#!/usr/bin/env python3
# 简单的输入文件检查脚本
import os
import yaml
from pathlib import Path

def main():
    print("🔍 EchoMimic 快速诊断")
    print("=" * 40)
    
    # 检查配置文件
    config_file = "./configs/prompts/infer.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请确认配置文件路径是否正确")
        return
    
    # 读取配置
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件读取成功")
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return
    
    # 打印配置内容
    print(f"\n📋 当前配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 检查关键文件
    files_to_check = [
        ('reference_image', '参考图像'),
        ('audio_path', '音频文件'),
        ('pose_dir', '姿势目录')
    ]
    
    print(f"\n🔍 检查输入文件:")
    all_good = True
    
    for key, name in files_to_check:
        if key not in config:
            print(f"❌ {name}: 配置中缺少 {key}")
            all_good = False
            continue
            
        path = config[key]
        print(f"\n📁 {name}: {path}")
        
        if os.path.exists(path):
            print(f"   ✅ 文件/目录存在")
            
            if key == 'reference_image':
                try:
                    from PIL import Image
                    img = Image.open(path)
                    print(f"   ✅ 图像尺寸: {img.size}")
                    print(f"   ✅ 图像格式: {img.format}")
                except Exception as e:
                    print(f"   ❌ 图像读取失败: {e}")
                    all_good = False
                    
            elif key == 'audio_path':
                file_size = os.path.getsize(path) / 1024
                print(f"   ✅ 文件大小: {file_size:.1f} KB")
                
            elif key == 'pose_dir':
                if os.path.isdir(path):
                    files = list(Path(path).glob("*"))
                    print(f"   ✅ 包含文件数: {len(files)}")
                    if files:
                        print(f"   ✅ 示例文件: {files[0].name}")
                else:
                    print(f"   ❌ 不是目录")
                    all_good = False
        else:
            print(f"   ❌ 文件/目录不存在")
            print(f"   💡 当前工作目录: {os.getcwd()}")
            print(f"   💡 绝对路径: {os.path.abspath(path)}")
            all_good = False
    
    print(f"\n{'🎉 所有检查通过!' if all_good else '❌ 存在问题，请修复后重试'}")
    
    if not all_good:
        print(f"\n💡 建议:")
        print(f"   1. 使用绝对路径而不是相对路径")
        print(f"   2. 确认文件确实存在于指定位置")
        print(f"   3. 检查文件权限")
        print(f"   4. 确认文件格式正确（图片用JPG/PNG，音频用WAV）")

if __name__ == "__main__":
    main()