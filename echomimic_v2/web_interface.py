import gradio as gr
import os
import sys
import subprocess
import time
from pathlib import Path

def generate_digital_human(reference_image, audio_file, pose_preset):
    """生成数字人视频的主函数"""
    
    try:
        status = "🎬 开始生成数字人视频...\n"
        
        # 检查输入文件
        if not reference_image:
            return None, status + "❌ 请选择参考图像"
        if not audio_file:
            return None, status + "❌ 请选择音频文件"
        
        status += f"📸 参考图像: {Path(reference_image).name}\n"
        status += f"🎵 音频文件: {Path(audio_file).name}\n"
        status += f"🤖 姿势预设: {pose_preset}\n\n"
        
        # 运行EchoMimic推理
        cmd = [sys.executable, "infer.py", "--config=./configs/prompts/infer.yaml"]
        
        status += "⚡ 正在生成视频，请稍候...\n"
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行推理
        result = subprocess.run(
            cmd, 
            cwd=str(Path.cwd()), 
            capture_output=True, 
            text=True, 
            timeout=600  # 10分钟超时
        )
        
        # 计算耗时
        duration = time.time() - start_time
        status += f"⏱️ 处理耗时: {duration:.1f}秒\n"
        
        # 查找生成的视频
        output_dir = Path("outputs")
        if output_dir.exists():
            video_files = list(output_dir.rglob("*.mp4"))
            if video_files:
                # 找最新的视频文件
                latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                status += f"✅ 视频生成成功！\n📁 位置: {latest_video}"
                return str(latest_video), status
        
        status += "❌ 未找到生成的视频文件"
        return None, status
        
    except subprocess.TimeoutExpired:
        return None, status + "❌ 生成超时（超过10分钟）"
    except Exception as e:
        return None, status + f"❌ 生成失败: {str(e)}"

def load_assets():
    """加载项目资源"""
    
    # 加载姿势预设
    pose_dir = Path("assets/halfbody_demo/pose")
    poses = []
    if pose_dir.exists():
        poses = [d.name for d in pose_dir.iterdir() if d.is_dir()]
    
    # 加载参考图像
    ref_dir = Path("assets/halfbody_demo/refimag")
    images = []
    if ref_dir.exists():
        for category in ref_dir.iterdir():
            if category.is_dir():
                for img in category.glob("*.png"):
                    images.append(str(img))
    
    # 加载音频文件
    audio_dir = Path("assets/halfbody_demo/audio") 
    audios = []
    if audio_dir.exists():
        for audio in audio_dir.rglob("*.wav"):
            audios.append(str(audio))
    
    return poses, images, audios

def create_interface():
    """创建Web界面"""
    
    # 加载资源
    poses, images, audios = load_assets()
    
    with gr.Blocks(title="EchoMimic数字人生成器") as app:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1 style="color: #2E8B57;">🎭 EchoMimic 数字人生成器</h1>
            <p style="color: #666; font-size: 16px;">用AI创建你的专属数字人视频</p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：输入控制
            with gr.Column(scale=1):
                gr.HTML("<h3>📋 生成设置</h3>")
                
                # 参考图像选择
                if images:
                    reference_image = gr.Dropdown(
                        choices=images,
                        label="📸 选择参考图像",
                        value=images[0]
                    )
                    img_preview = gr.Image(
                        label="图像预览", 
                        interactive=False,
                        height=150
                    )
                else:
                    reference_image = gr.Textbox(label="参考图像路径")
                    img_preview = gr.HTML("<p>❌ 未找到参考图像</p>")
                
                # 音频文件选择  
                if audios:
                    audio_file = gr.Dropdown(
                        choices=audios,
                        label="🎵 选择音频文件", 
                        value=audios[0]
                    )
                    audio_preview = gr.Audio(
                        label="音频预览",
                        interactive=False
                    )
                else:
                    audio_file = gr.Textbox(label="音频文件路径")
                    audio_preview = gr.HTML("<p>❌ 未找到音频文件</p>")
                
                # 姿势预设
                if poses:
                    pose_preset = gr.Dropdown(
                        choices=poses,
                        label="🤖 选择姿势预设",
                        value=poses[0]
                    )
                else:
                    pose_preset = gr.Dropdown(
                        choices=["01"],
                        label="🤖 姿势预设", 
                        value="01"
                    )
                
                # 生成按钮
                generate_btn = gr.Button(
                    "🎬 开始生成",
                    variant="primary",
                    size="lg"
                )
            
            # 右侧：输出结果
            with gr.Column(scale=1):
                gr.HTML("<h3>📺 生成结果</h3>")
                
                output_video = gr.Video(
                    label="生成的数字人视频",
                    height=300
                )
                
                status_text = gr.Textbox(
                    label="状态信息",
                    lines=8,
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
        
        # 预览更新
        if images:
            reference_image.change(
                update_image_preview,
                inputs=[reference_image],
                outputs=[img_preview]
            )
        
        if audios:
            audio_file.change(
                update_audio_preview, 
                inputs=[audio_file],
                outputs=[audio_preview]
            )
        
        # 生成事件
        generate_btn.click(
            generate_digital_human,
            inputs=[reference_image, audio_file, pose_preset],
            outputs=[output_video, status_text]
        )
    
    return app

def main():
    """启动Web应用"""
    
    print("🚀 启动EchoMimic Web界面...")
    
    # 检查环境
    if not Path("infer.py").exists():
        print("❌ 错误：找不到infer.py，请在EchoMimic项目根目录运行")
        return
    
    try:
        app = create_interface()
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            share=False
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()