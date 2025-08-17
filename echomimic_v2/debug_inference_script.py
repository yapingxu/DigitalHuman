#!/usr/bin/env python3
# 推理脚本调试检查工具
# 检查推理脚本是否正确读取和使用配置文件

import os
import sys
import yaml
import argparse
from pathlib import Path

def check_config_loading():
    """检查配置文件加载逻辑"""
    
    print("🔍 检查推理脚本的配置加载...")
    
    # 检查是否存在推理脚本
    infer_script = Path("infer.py")
    if not infer_script.exists():
        print("❌ infer.py 不存在！")
        return False
    
    # 读取推理脚本内容
    try:
        with open(infer_script, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        print("✅ 推理脚本读取成功")
        
        # 检查关键配置读取逻辑
        checks = {
            "yaml.safe_load": "yaml.safe_load" in script_content,
            "reference_image": "reference_image" in script_content,
            "audio_path": "audio_path" in script_content,
            "pose_dir": "pose_dir" in script_content,
            "argparse": "argparse" in script_content,
            "config参数": "--config" in script_content,
        }
        
        print("\n📋 推理脚本内容检查:")
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check}: {'找到' if result else '未找到'}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"❌ 读取推理脚本失败: {e}")
        return False

def simulate_config_loading(config_path):
    """模拟推理脚本加载配置的过程"""
    
    print(f"\n🔧 模拟配置加载过程...")
    print(f"配置文件: {config_path}")
    
    if not Path(config_path).exists():
        print("❌ 配置文件不存在")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        print(f"✅ 配置内容: {len(config)} 个参数")
        
        # 检查关键参数
        key_params = ['reference_image', 'audio_path', 'pose_dir']
        
        print(f"\n📸 关键参数检查:")
        for param in key_params:
            if param in config:
                value = config[param]
                exists = Path(value).exists() if value else False
                print(f"   ✅ {param}: {value}")
                print(f"      {'✅' if exists else '❌'} 文件存在: {exists}")
            else:
                print(f"   ❌ {param}: 未找到")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return None

def check_inference_pipeline_code():
    """检查推理管道代码"""
    
    print(f"\n🔬 检查推理管道代码...")
    
    infer_script = Path("infer.py")
    
    try:
        with open(infer_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有硬编码的默认路径
        problematic_patterns = [
            "assets/halfbody_demo/refimag",  # 硬编码的参考图像
            "assets/halfbody_demo/audio",    # 硬编码的音频
            ".jpg",                          # 可能硬编码的图片
            ".wav",                          # 可能硬编码的音频
        ]
        
        print("🚨 检查可能的硬编码问题:")
        for pattern in problematic_patterns:
            if pattern in content:
                print(f"   ⚠️ 发现可疑硬编码: '{pattern}'")
                # 显示相关代码行
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line:
                        print(f"      第{i+1}行: {line.strip()}")
        
        # 检查配置使用逻辑
        config_usage_patterns = [
            "config.get('reference_image')",
            "config['reference_image']",
            "config.get('audio_path')",
            "config['audio_path']",
        ]
        
        print(f"\n✅ 检查配置使用:")
        found_usage = False
        for pattern in config_usage_patterns:
            if pattern in content:
                print(f"   ✅ 找到: {pattern}")
                found_usage = True
        
        if not found_usage:
            print("   ❌ 未找到配置参数的使用代码！这可能是问题所在！")
        
        return found_usage
        
    except Exception as e:
        print(f"❌ 检查推理脚本失败: {e}")
        return False

def create_minimal_test_config():
    """创建最小测试配置"""
    
    print(f"\n🔧 创建测试配置...")
    
    # 找到最近的动态配置文件
    temp_dir = Path("temp_uploads")
    if temp_dir.exists():
        config_files = list(temp_dir.glob("dynamic_config_*.yaml"))
        if config_files:
            latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 找到最新配置: {latest_config}")
            
            # 读取并显示内容
            with open(latest_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"📋 配置内容:")
            for key, value in config.items():
                if key in ['reference_image', 'audio_path', 'pose_dir']:
                    exists = Path(value).exists() if isinstance(value, str) else False
                    print(f"   {key}: {value} ({'✅' if exists else '❌'})")
            
            return str(latest_config)
    
    return None

def check_echomimic_pipeline():
    """检查EchoMimic管道实现"""
    
    print(f"\n🔍 检查EchoMimic管道实现...")
    
    # 检查管道文件
    pipeline_files = [
        "src/pipelines/pipeline_echomimic.py",
        "pipelines/pipeline_echomimic.py", 
        "echomimic/pipeline.py",
        "pipeline_echomimic.py"
    ]
    
    pipeline_file = None
    for pf in pipeline_files:
        if Path(pf).exists():
            pipeline_file = Path(pf)
            break
    
    if not pipeline_file:
        print("❌ 未找到EchoMimic管道文件")
        return False
    
    print(f"✅ 找到管道文件: {pipeline_file}")
    
    try:
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查管道的__call__方法
        if "def __call__" in content:
            print("✅ 找到__call__方法")
            
            # 检查参数处理
            param_checks = [
                "reference_image",
                "audio_path", 
                "pose_dir"
            ]
            
            print("📋 参数处理检查:")
            for param in param_checks:
                if param in content:
                    print(f"   ✅ 处理参数: {param}")
                else:
                    print(f"   ❌ 未找到参数处理: {param}")
        else:
            print("❌ 未找到__call__方法")
            
        return True
        
    except Exception as e:
        print(f"❌ 检查管道文件失败: {e}")
        return False

def main():
    print("🔍 EchoMimic 推理脚本深度诊断")
    print("🎯 找出为什么不使用上传的文件")
    print("=" * 50)
    
    # 检查1: 配置加载逻辑
    config_ok = check_config_loading()
    
    # 检查2: 推理管道代码
    pipeline_ok = check_inference_pipeline_code()
    
    # 检查3: 测试配置文件
    test_config = create_minimal_test_config()
    if test_config:
        simulate_config_loading(test_config)
    
    # 检查4: EchoMimic管道
    echomimic_ok = check_echomimic_pipeline()
    
    print(f"\n📊 诊断结果汇总:")
    print(f"   配置加载: {'✅' if config_ok else '❌'}")
    print(f"   管道代码: {'✅' if pipeline_ok else '❌'}")
    print(f"   测试配置: {'✅' if test_config else '❌'}")
    print(f"   管道实现: {'✅' if echomimic_ok else '❌'}")
    
    if not all([config_ok, pipeline_ok, echomimic_ok]):
        print(f"\n🚨 发现问题！")
        print(f"💡 建议:")
        
        if not config_ok:
            print(f"   - 推理脚本可能没有正确的配置加载逻辑")
            
        if not pipeline_ok:
            print(f"   - 推理脚本可能有硬编码路径")
            
        if not echomimic_ok:
            print(f"   - EchoMimic管道可能不支持自定义输入")
    else:
        print(f"\n🤔 代码逻辑看起来正常，问题可能在:")
        print(f"   - 模型权重文件本身的限制")
        print(f"   - 管道内部的默认行为")
        print(f"   - 某些隐含的缓存机制")

if __name__ == "__main__":
    main()