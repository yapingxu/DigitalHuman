import gradio as gr
import os
import sys
import torch
import subprocess
import time
from pathlib import Path

def generate_video_simple(reference_image, audio_file, pose_preset):
    """简化的视频生成函数"""
    
    try:
        # 基本信息
        status_msg = f"🎬 开始生成数字人视频...\n"
        status_msg += f"📸 参考图像: {Path(reference_image).name if reference_image else '未选择'}\n"
        status_msg += f"🎵 音频文件: {Path(audio_file).name if audio_file else '未选择'}\n"
        status_msg += f"🤖 姿势预设: {pose_preset}\n"
        
        # 检查输入
        if not reference_image or not os.path.exists(reference_image):
            return None, status_msg + "❌ 错误: 请选择有效的参考图像"
        
        if not audio_file or not os.path.exists(audio_file):
            return None, status_msg + "❌ 错误: 请选择有效的音频文件"
        
        # 构建推理命令 - 使用原始的infer.py
        cmd = [
            sys.executable, 
            "infer.py", 
            "--config=./configs/prompts/infer.yaml"
        ]
        
        status_msg += "⚡ 启动推理进程...\n"
        
        # 运行推理
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            duration = time.time() - start_time
            status_msg += f"⏱️ 处理耗时: {duration:.1f}秒\n"
            
            if result.returncode == 0:
                status_msg += "✅ 推理完成!\n"
                
                # 查找生成的视频文件
                output_dir = Path("outputs")
                video_files = list(output_dir.rglob("*.mp4"))
                
                if video_files:
                    # 找到最新的视频文件
                    latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                    status_msg += f"📁 视频保存位置: {latest_video}\n"
                    return str(latest_video), status_msg
                else:
                    status_msg += "❌ 未找到生成的视频文件\n"
                    return None, status_msg
                    
            else:
                status_msg += f"⚠️ 推理完成但有警告 (返回码: {result.returncode})\n"
                status_msg += f"错误信息:\n{result.stderr}\n"
                
                # 即使有警告，也尝试查找视频文件
                output_dir = Path("outputs")
                video_files = list(output_dir.rglob("*.mp4"))
                
                if video_files:
                    latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                    status_msg += f"📁 找到视频文件: {latest_video}\n"
                    return str(latest_video), status_msg
                else:
                    return None, status_msg
                
        except subprocess.TimeoutExpired:
            status_msg += "❌ 处理超时 (超过10分钟)\n"
            return None, status_msg
            
    except Exception as e:
        status_msg += f"❌ 生成失败: {str(e)}\n"
        import traceback
        status_msg += f"详细错误:\n{traceback.format_exc()}\n"
        return None, status_msg

def create_simple_interface():
    """创建简化的Gradio界面"""
    
    # 获取预设资源
    pose_dir = Path("assets/halfbody_demo/pose")
    pose_choices = [d.name for d in pose_dir.iterdir() if d.is_dir()] if pose_dir.exists() else ["01"]
    
    ref_dir = Path("assets/halfbody_demo/refimag")
    ref_images = []
    if ref_dir.exists():
        for category in ref_dir.iterdir():
            if category.is_dir():
                for img in category.glob("*.png"):
                    ref_images.append(str(img))
    
    audio_dir = Path("assets/halfbody_demo/audio")
    audio_files = []
    if audio_dir.exists():
        for audio in audio_dir.rglob("*.wav"):
            audio_files.append(str(audio))
    
    with gr.Blocks(title="EchoMimic 数字人生成器") as interface:
        
        gr.HTML("""
        <h1 style="text-align: center; color: #2E8B57;">🎭 EchoMimic 数字人生成器 (测试版)</h1>
        <p style="text-align: center; color: #666;">简化版界面，用于测试基本功能</p>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>📸 选择参考图像</h3>")
                
                if ref_images:
                    reference_image = gr.Dropdown(
                        choices=ref_images,
                        label="参考图像",
                        value=ref_images[0] if ref_images else None
                    )
                    
                    image_preview = gr.Image(
                        label="图像预览",
                        interactive=False,
                        height=200
                    )
                else:
                    reference_image = gr.Textbox(
                        label="参考图像路径",
                        placeholder="请输入图像路径"
                    )
                    image_preview = None
                
                gr.HTML("<h3>🎵 选择音频文件</h3>")
                
                if audio_files:
                    audio_file = gr.Dropdown(
                        choices=audio_files,
                        label="音频文件",
                        value=audio_files[0] if audio_files else None
                    )
                    
                    audio_preview = gr.Audio(
                        label="音频预览",
                        interactive=False
                    )
                else:
                    audio_file = gr.Textbox(
                        label="音频文件路径",
                        placeholder="请输入音频路径"
                    )
                    audio_preview = None
            
            with gr.Column():
                gr.HTML("<h3>🤖 设置姿势</h3>")
                
                pose_preset = gr.Dropdown(
                    choices=pose_choices,
                    label="姿势预设",
                    value=pose_choices[0] if pose_choices else "01"
                )
                
                generate_btn = gr.Button(
                    "🎬 生成数字人视频",
                    variant="primary",
                    size="lg"
                )
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>📺 生成结果</h3>")
                
                output_video = gr.Video(
                    label="生成的视频",
                    height=400
                )
                
                status_output = gr.Textbox(
                    label="状态信息",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
        
        # 事件绑定
        def update_image_preview(img_path):
            if img_path and os.path.exists(img_path):
                return img_path
            return None
        
        def update_audio_preview(audio_path):
            if audio_path and os.path.exists(audio_path):
                return audio_path
            return None
        
        if ref_images and image_preview:
            reference_image.change(
                update_image_preview,
                inputs=[reference_image],
                outputs=[image_preview]
            )
        
        if audio_files and audio_preview:
            audio_file.change(
                update_audio_preview,
                inputs=[audio_file],
                outputs=[audio_preview]
            )
        
        generate_btn.click(
            generate_video_simple,
            inputs=[reference_image, audio_file, pose_preset],
            outputs=[output_video, status_output]
        )
    
    return interface

def main():
    """主函数"""
    print("🎭 EchoMimic 简化Web界面")
    print("=" * 30)
    
    # 检查基本环境
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    try:
        import gradio as gr
        print(f"✅ Gradio: {gr.__version__}")
    except ImportError:
        print("❌ Gradio未安装")
        return
    
    # 检查项目文件
    if not Path("infer.py").exists():
        print("❌ 找不到 infer.py，请确保在项目根目录运行")
        return
    
    print("🚀 启动Web界面...")
    
    try:
        interface = create_simple_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            debug=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()