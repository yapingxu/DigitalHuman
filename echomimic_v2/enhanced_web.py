import gradio as gr
import os
import sys
import subprocess
import time
import shutil
import yaml
from pathlib import Path
from PIL import Image
import librosa
import soundfile as sf

class EchoMimicApp:
    def __init__(self):
        self.project_root = Path.cwd()
        self.temp_dir = self.project_root / "temp_uploads"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 加载项目资源
        self.poses, self.images, self.audios = self.load_assets()
    
    def load_assets(self):
        """加载项目资源"""
        
        # 加载姿势预设
        pose_dir = self.project_root / "assets/halfbody_demo/pose"
        poses = []
        if pose_dir.exists():
            poses = [d.name for d in pose_dir.iterdir() if d.is_dir()]
        
        # 加载参考图像
        ref_dir = self.project_root / "assets/halfbody_demo/refimag"
        images = []
        if ref_dir.exists():
            for category in ref_dir.iterdir():
                if category.is_dir():
                    for img in category.glob("*.png"):
                        images.append(str(img))
        
        # 加载音频文件
        audio_dir = self.project_root / "assets/halfbody_demo/audio" 
        audios = []
        if audio_dir.exists():
            for audio in audio_dir.rglob("*.wav"):
                audios.append(str(audio))
        
        return poses, images, audios
    
    def preprocess_custom_image(self, image_file, target_size=(512, 512)):
        """处理用户上传的自定义图像"""
        
        if not image_file:
            return None, "❌ 未上传图像文件"
        
        try:
            # 打开图像
            img = Image.open(image_file.name)
            
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 获取原始尺寸
            original_size = img.size
            
            # 计算缩放比例，保持长宽比
            aspect_ratio = original_size[0] / original_size[1]
            
            if aspect_ratio > 1:  # 宽图
                new_width = target_size[0]
                new_height = int(target_size[0] / aspect_ratio)
            else:  # 高图或正方形
                new_height = target_size[1]
                new_width = int(target_size[1] * aspect_ratio)
            
            # 缩放图像
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建目标尺寸的黑色背景
            canvas = Image.new('RGB', target_size, (0, 0, 0))
            
            # 居中粘贴
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            canvas.paste(img, (paste_x, paste_y))
            
            # 保存处理后的图像
            processed_path = self.temp_dir / f"custom_image_{int(time.time())}.png"
            canvas.save(processed_path)
            
            return str(processed_path), f"✅ 图像处理成功\n原始尺寸: {original_size[0]}x{original_size[1]}\n处理后: {target_size[0]}x{target_size[1]}"
            
        except Exception as e:
            return None, f"❌ 图像处理失败: {str(e)}"
    
    def preprocess_custom_audio(self, audio_file, target_sr=16000, max_duration=30):
        """处理用户上传的自定义音频"""
        
        if not audio_file:
            return None, "❌ 未上传音频文件"
        
        try:
            # 加载音频
            audio, sr = librosa.load(audio_file.name, sr=None)
            original_duration = len(audio) / sr
            
            # 限制音频长度
            if original_duration > max_duration:
                audio = audio[:int(max_duration * sr)]
                duration_info = f"⚠️ 音频已截断至{max_duration}秒"
            else:
                duration_info = f"✅ 音频长度: {original_duration:.2f}秒"
            
            # 重采样
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # 归一化
            audio = librosa.util.normalize(audio)
            
            # 保存处理后的音频
            processed_path = self.temp_dir / f"custom_audio_{int(time.time())}.wav"
            sf.write(processed_path, audio, sr)
            
            return str(processed_path), f"{duration_info}\n采样率: {sr}Hz\n✅ 音频处理完成"
            
        except Exception as e:
            return None, f"❌ 音频处理失败: {str(e)}"
    
    def create_temp_config(self, ref_image_path, audio_path, pose_preset):
        """创建临时配置文件"""
        
        pose_path = self.project_root / "assets/halfbody_demo/pose" / pose_preset
        
        config = {
            'reference_image': ref_image_path,
            'audio_path': audio_path,
            'pose_dir': str(pose_path),
            'output_dir': str(self.project_root / "outputs")
        }
        
        config_path = self.temp_dir / f"temp_config_{int(time.time())}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return str(config_path)
    
    def generate_digital_human(self, use_custom_image, custom_image, preset_image, 
                             use_custom_audio, custom_audio, preset_audio, pose_preset):
        """生成数字人视频"""
        
        try:
            status = "🎬 开始生成数字人视频...\n"
            
            # 处理图像输入
            if use_custom_image and custom_image:
                status += "📸 处理自定义图像...\n"
                ref_image_path, img_msg = self.preprocess_custom_image(custom_image)
                if not ref_image_path:
                    return None, status + img_msg
                status += img_msg + "\n\n"
            else:
                if not preset_image:
                    return None, status + "❌ 请选择预设图像"
                ref_image_path = preset_image
                status += f"📸 使用预设图像: {Path(preset_image).name}\n\n"
            
            # 处理音频输入
            if use_custom_audio and custom_audio:
                status += "🎵 处理自定义音频...\n"
                audio_path, audio_msg = self.preprocess_custom_audio(custom_audio)
                if not audio_path:
                    return None, status + audio_msg
                status += audio_msg + "\n\n"
            else:
                if not preset_audio:
                    return None, status + "❌ 请选择预设音频"
                audio_path = preset_audio
                status += f"🎵 使用预设音频: {Path(preset_audio).name}\n\n"
            
            status += f"🤖 使用姿势预设: {pose_preset}\n\n"
            
            # 创建临时配置
            # config_path = self.create_temp_config(ref_image_path, audio_path, pose_preset)
            
            # 运行EchoMimic推理
            cmd = [sys.executable, "infer.py", "--config=./configs/prompts/infer.yaml"]
            
            status += "⚡ 正在生成视频，这可能需要几分钟...\n"
            
            start_time = time.time()
            
            # 执行推理
            result = subprocess.run(
                cmd, 
                cwd=str(self.project_root), 
                capture_output=True, 
                text=True, 
                timeout=900  # 15分钟超时
            )
            
            duration = time.time() - start_time
            status += f"⏱️ 处理耗时: {duration:.1f}秒\n"
            
            # 查找生成的视频
            output_dir = self.project_root / "outputs"
            if output_dir.exists():
                video_files = list(output_dir.rglob("*.mp4"))
                if video_files:
                    latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                    
                    # 如果使用了自定义内容，重命名视频文件以便识别
                    if use_custom_image or use_custom_audio:
                        custom_name = "custom_"
                        if use_custom_image:
                            custom_name += "img_"
                        if use_custom_audio:
                            custom_name += "audio_"
                        custom_name += f"{int(time.time())}.mp4"
                        
                        new_path = latest_video.parent / custom_name
                        shutil.copy2(latest_video, new_path)
                        latest_video = new_path
                    
                    status += f"✅ 视频生成成功！\n📁 保存位置: {latest_video}"
                    return str(latest_video), status
            
            status += "❌ 未找到生成的视频文件"
            return None, status
            
        except subprocess.TimeoutExpired:
            return None, status + "❌ 生成超时（超过15分钟）"
        except Exception as e:
            return None, status + f"❌ 生成失败: {str(e)}"
    
    def create_interface(self):
        """创建Web界面"""
        
        with gr.Blocks(title="EchoMimic数字人生成器", theme=gr.themes.Soft()) as app:
            
            gr.HTML("""
            <div style="text-align: center; margin: 30px;">
                <h1 style="color: #2E8B57; font-size: 2.5em;">🎭 EchoMimic 数字人生成器</h1>
                <p style="color: #666; font-size: 18px;">创建属于你的个性化数字人视频</p>
                <p style="color: #999; font-size: 14px;">支持自定义图像和音频，让数字人更像你！</p>
            </div>
            """)
            
            with gr.Row():
                # 左侧：图像设置
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #4CAF50;'>📸 图像设置</h3>")
                    
                    use_custom_image = gr.Checkbox(
                        label="使用我的照片",
                        value=False
                    )
                    
                    with gr.Group() as preset_img_group:
                        if self.images:
                            preset_image = gr.Dropdown(
                                choices=self.images,
                                label="选择预设图像",
                                value=self.images[0]
                            )
                            preset_img_preview = gr.Image(
                                label="预设图像预览", 
                                interactive=False,
                                height=200
                            )
                        else:
                            preset_image = gr.HTML("<p>❌ 未找到预设图像</p>")
                            preset_img_preview = None
                    
                    with gr.Group(visible=False) as custom_img_group:
                        custom_image = gr.File(
                            label="上传你的照片",
                            file_types=[".jpg", ".jpeg", ".png", ".bmp"]
                        )
                        custom_img_preview = gr.Image(
                            label="自定义图像预览",
                            interactive=False,
                            height=200
                        )
                        gr.HTML("<p style='color: #666; font-size: 12px;'>💡 建议使用清晰的正面照片，支持JPG/PNG格式</p>")
                
                # 中间：音频设置
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #4CAF50;'>🎵 音频设置</h3>")
                    
                    use_custom_audio = gr.Checkbox(
                        label="使用我的声音",
                        value=False
                    )
                    
                    with gr.Group() as preset_audio_group:
                        if self.audios:
                            preset_audio = gr.Dropdown(
                                choices=self.audios,
                                label="选择预设音频",
                                value=self.audios[0]
                            )
                            preset_audio_preview = gr.Audio(
                                label="预设音频试听",
                                interactive=False
                            )
                        else:
                            preset_audio = gr.HTML("<p>❌ 未找到预设音频</p>")
                            preset_audio_preview = None
                    
                    with gr.Group(visible=False) as custom_audio_group:
                        custom_audio = gr.File(
                            label="上传你的音频",
                            file_types=[".wav", ".mp3", ".m4a", ".flac"]
                        )
                        custom_audio_preview = gr.Audio(
                            label="自定义音频试听",
                            interactive=False
                        )
                        gr.HTML("<p style='color: #666; font-size: 12px;'>💡 建议时长10-30秒，支持WAV/MP3格式</p>")
                
                # 右侧：其他设置
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #4CAF50;'>🤖 姿势设置</h3>")
                    
                    if self.poses:
                        pose_preset = gr.Dropdown(
                            choices=self.poses,
                            label="选择姿势预设",
                            value=self.poses[0]
                        )
                    else:
                        pose_preset = gr.Dropdown(
                            choices=["01"],
                            label="姿势预设", 
                            value="01"
                        )
                    
                    gr.HTML("<br>")
                    
                    generate_btn = gr.Button(
                        "🎬 生成我的数字人",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.HTML("""
                    <div style="margin-top: 20px; padding: 10px; background-color: #f0f8f0; border-radius: 5px;">
                        <h4 style="color: #2E8B57;">✨ 个性化提示</h4>
                        <ul style="color: #666; font-size: 12px;">
                            <li>上传你的照片，让数字人拥有你的外貌</li>
                            <li>录制你的声音，让数字人说出你的话</li>
                            <li>选择不同姿势，创造独特的表演效果</li>
                        </ul>
                    </div>
                    """)
            
            # 输出区域
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3 style='color: #4CAF50;'>📺 生成结果</h3>")
                    
                    output_video = gr.Video(
                        label="你的专属数字人视频",
                        height=400
                    )
                    
                    status_text = gr.Textbox(
                        label="生成状态",
                        lines=10,
                        interactive=False
                    )
            
            # 事件处理
            def toggle_image_input(use_custom):
                return gr.Group.update(visible=not use_custom), gr.Group.update(visible=use_custom)
            
            def toggle_audio_input(use_custom):
                return gr.Group.update(visible=not use_custom), gr.Group.update(visible=use_custom)
            
            def update_preset_img_preview(img_path):
                return img_path if img_path and os.path.exists(img_path) else None
            
            def update_custom_img_preview(img_file):
                return img_file.name if img_file else None
            
            def update_preset_audio_preview(audio_path):
                return audio_path if audio_path and os.path.exists(audio_path) else None
            
            def update_custom_audio_preview(audio_file):
                return audio_file.name if audio_file else None
            
            # 绑定事件
            use_custom_image.change(
                toggle_image_input,
                inputs=[use_custom_image],
                outputs=[preset_img_group, custom_img_group]
            )
            
            use_custom_audio.change(
                toggle_audio_input,
                inputs=[use_custom_audio],
                outputs=[preset_audio_group, custom_audio_group]
            )
            
            if self.images and preset_img_preview:
                preset_image.change(
                    update_preset_img_preview,
                    inputs=[preset_image],
                    outputs=[preset_img_preview]
                )
            
            custom_image.change(
                update_custom_img_preview,
                inputs=[custom_image],
                outputs=[custom_img_preview]
            )
            
            if self.audios and preset_audio_preview:
                preset_audio.change(
                    update_preset_audio_preview,
                    inputs=[preset_audio],
                    outputs=[preset_audio_preview]
                )
            
            custom_audio.change(
                update_custom_audio_preview,
                inputs=[custom_audio],
                outputs=[custom_audio_preview]
            )
            
            # 生成按钮事件
            generate_btn.click(
                self.generate_digital_human,
                inputs=[
                    use_custom_image, custom_image, preset_image,
                    use_custom_audio, custom_audio, preset_audio, 
                    pose_preset
                ],
                outputs=[output_video, status_text]
            )
        
        return app

def main():
    """启动应用"""
    
    print("🚀 启动EchoMimic增强版Web界面...")
    
    # 检查环境
    if not Path("infer.py").exists():
        print("❌ 错误：找不到infer.py，请在EchoMimic项目根目录运行")
        return
    
    try:
        app = EchoMimicApp()
        interface = app.create_interface()
        
        print("🌐 Web界面已启动")
        print("📍 访问地址: http://127.0.0.1:7860")
        print("💡 现在支持上传你的照片和音频文件了！")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            share=False
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()