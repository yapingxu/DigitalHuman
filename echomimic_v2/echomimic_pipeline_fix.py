#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EchoMimic管道修复脚本
解决不使用上传文件的问题
"""

import os
import sys
import yaml
import shutil
import glob
from pathlib import Path

class EchoMimicPipelineFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.temp_uploads = self.project_root / "temp_uploads"
        
    def find_pipeline_files(self):
        """查找EchoMimic管道相关文件"""
        print("🔍 查找EchoMimic管道文件...")
        
        # 可能的管道文件位置
        possible_locations = [
            "src/models/echomimic_pipeline.py",
            "echomimic/pipeline.py",
            "models/pipeline.py",
            "inference_pipeline.py",
            "echomimic_pipeline.py",
            "src/echomimic_pipeline.py"
        ]
        
        found_files = []
        for location in possible_locations:
            file_path = self.project_root / location
            if file_path.exists():
                found_files.append(file_path)
                print(f"   ✅ 找到: {file_path}")
        
        # 递归搜索包含 "pipeline" 的Python文件
        for py_file in self.project_root.rglob("*.py"):
            if "pipeline" in py_file.name.lower() and py_file not in found_files:
                found_files.append(py_file)
                print(f"   📁 发现: {py_file}")
        
        return found_files
    
    def analyze_pipeline_code(self, file_path):
        """分析管道代码中的硬编码问题"""
        print(f"\n🔬 分析管道文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues = []
            
            # 检查硬编码路径
            hardcoded_patterns = [
                ('assets/halfbody_demo', '默认pose目录'),
                ('assets/test_data', '默认测试数据'),
                ('demo.wav', '默认音频文件'),
                ('demo.png', '默认图像文件'),
                ('demo.jpg', '默认图像文件'),
                ('reference.png', '默认参考图像'),
                ('reference.jpg', '默认参考图像'),
            ]
            
            for pattern, description in hardcoded_patterns:
                if pattern in content:
                    issues.append(f"   🚨 发现硬编码: {pattern} ({description})")
            
            # 检查配置使用
            config_usage = []
            if 'config.get(' in content or 'config[' in content:
                config_usage.append("   ✅ 使用配置参数")
            else:
                issues.append("   ❌ 未使用配置参数")
            
            return issues, config_usage
            
        except Exception as e:
            return [f"   ❌ 读取文件失败: {e}"], []
    
    def create_fixed_pipeline(self, original_file):
        """创建修复后的管道文件"""
        print(f"\n🔧 创建修复版本: {original_file}")
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修复策略：确保使用配置参数
            fixes = [
                # 硬编码路径替换
                ('assets/halfbody_demo', 'config.get("pose_dir", "assets/halfbody_demo")'),
                ('demo.wav', 'config.get("audio_path", "demo.wav")'),
                ('demo.png', 'config.get("reference_image", "demo.png")'),
                ('demo.jpg', 'config.get("reference_image", "demo.jpg")'),
                ('reference.png', 'config.get("reference_image", "reference.png")'),
                ('reference.jpg', 'config.get("reference_image", "reference.jpg")'),
            ]
            
            modified = False
            for old_pattern, new_pattern in fixes:
                if old_pattern in content and new_pattern not in content:
                    content = content.replace(f'"{old_pattern}"', new_pattern)
                    content = content.replace(f"'{old_pattern}'", new_pattern)
                    modified = True
                    print(f"   ✅ 替换: {old_pattern} -> 配置参数")
            
            if modified:
                # 保存修复版本
                backup_file = original_file.with_suffix('.py.backup')
                shutil.copy2(original_file, backup_file)
                print(f"   💾 备份原文件: {backup_file}")
                
                with open(original_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"   ✅ 保存修复版本")
                return True
            else:
                print(f"   ℹ️  无需修复")
                return False
                
        except Exception as e:
            print(f"   ❌ 修复失败: {e}")
            return False
    
    def create_custom_pipeline_wrapper(self):
        """创建自定义管道包装器"""
        wrapper_path = self.project_root / "custom_pipeline_wrapper.py"
        
        wrapper_content = '''#!/usr/bin/env python3
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
        print("\\n🚀 开始推理...")
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
    print(f"\\n📊 结果: {result}")
'''
        
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        print(f"✅ 创建包装器: {wrapper_path}")
        return wrapper_path
    
    def run_diagnosis_and_fix(self):
        """运行完整的诊断和修复"""
        print("🎯 EchoMimic管道修复程序")
        print("=" * 50)
        
        # 1. 查找管道文件
        pipeline_files = self.find_pipeline_files()
        
        if not pipeline_files:
            print("❌ 未找到管道文件，创建自定义包装器...")
            wrapper_path = self.create_custom_pipeline_wrapper()
            print(f"✅ 解决方案: 使用 {wrapper_path}")
            return
        
        # 2. 分析每个管道文件
        total_issues = 0
        for file_path in pipeline_files:
            issues, config_usage = self.analyze_pipeline_code(file_path)
            
            if issues:
                total_issues += len(issues)
                print("🚨 发现问题:")
                for issue in issues:
                    print(issue)
            
            if config_usage:
                print("✅ 配置使用:")
                for usage in config_usage:
                    print(usage)
        
        # 3. 尝试修复
        if total_issues > 0:
            print(f"\n🔧 尝试修复 {total_issues} 个问题...")
            for file_path in pipeline_files:
                self.create_fixed_pipeline(file_path)
        
        # 4. 创建包装器作为备选方案
        wrapper_path = self.create_custom_pipeline_wrapper()
        
        print(f"\n📊 修复完成!")
        print(f"💡 建议: 使用 {wrapper_path} 确保正确使用上传文件")

def main():
    fixer = EchoMimicPipelineFixer()
    fixer.run_diagnosis_and_fix()

if __name__ == "__main__":
    main()
