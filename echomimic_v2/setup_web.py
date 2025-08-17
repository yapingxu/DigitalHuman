import subprocess
import sys
import os

def install_web_dependencies():
    """安装Web界面所需的依赖"""
    
    dependencies = [
        'gradio>=3.50.0',
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'opencv-python>=4.8.0',
        'pillow>=9.5.0',
        'pyyaml>=6.0',
        'numpy>=1.24.0'
    ]
    
    print("🔧 开始安装Web界面依赖...")
    
    for dep in dependencies:
        print(f"安装 {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dep} 安装失败: {e}")
            return False
    
    print("🎉 所有依赖安装完成!")
    return True

def create_directory_structure():
    """创建必要的目录结构"""
    
    directories = [
        "temp_uploads",
        "outputs",
        "assets/custom_images",
        "assets/custom_audios"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 创建目录: {directory}")

def check_system_requirements():
    """检查系统要求"""
    
    print("🔍 检查系统要求...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️ CUDA不可用，将使用CPU（生成速度较慢）")
    
    except ImportError:
        print("❌ PyTorch未安装，请先安装PyTorch")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 EchoMimic Web界面安装程序")
    print("=" * 50)
    
    if not check_system_requirements():
        print("❌ 系统要求不满足，请先安装必要的依赖")
        sys.exit(1)
    
    create_directory_structure()
    
    if install_web_dependencies():
        print("\n🎉 Web界面安装完成!")
        print("现在可以运行: python web_app.py")
    else:
        print("\n❌ 安装过程中出现错误")
        sys.exit(1)