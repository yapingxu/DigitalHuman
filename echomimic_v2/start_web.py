import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查是否已安装必要的依赖"""
    
    required_modules = [
        'gradio', 'librosa', 'soundfile', 'cv2', 'PIL', 'yaml'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'cv2':
                __import__('cv2')
            elif module == 'PIL':
                __import__('PIL')
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ 缺少依赖: {', '.join(missing_modules)}")
        print("请先运行: python setup_web.py")
        return False
    
    return True

def start_web_interface():
    """启动Web界面"""
    
    print("🌐 启动EchoMimic Web界面...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查项目结构
    required_paths = [
        "assets/halfbody_demo",
        "configs/prompts",
        "infer.py"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ 缺少必要文件/目录: {', '.join(missing_paths)}")
        print("请确保在EchoMimic项目根目录下运行此脚本")
        return
    
    # 启动Web应用
    try:
        print("🚀 正在启动Web服务器...")
        print("📍 访问地址: http://127.0.0.1:7860")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("-" * 50)
        
        # 延时后打开浏览器
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://127.0.0.1:7860")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # 运行Web应用
        subprocess.run([sys.executable, "web_app.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    start_web_interface()