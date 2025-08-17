import gradio as gr
import os
import sys
import torch
import subprocess
import shutil
import json
import time
from pathlib import Path
import yaml
import cv2
import numpy as np
from PIL import Image
import librosa
import soundfile as sf

class EchoMimicWebApp:
    def __init__(self):
        self.project_root = Path("D:/PythonProjects/DigitalHuman/echomimic_v2")
        self.output_dir = self.project_root / "outputs"
        self.temp_dir = self.project_root / "temp_uploads"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 预设姿势和参考图像路径
        self.pose_presets = self._load_pose_presets()
        self.reference_images = self._load_reference_images()
        
    def _load_pose_presets(self):
        """加载预设姿势"""
        pose_dir = self.project_root / "assets" / "halfbody_demo" / "pose"
        presets = {}
        if pose_dir.exists():
            for pose_folder in pose_dir.iterdir():
                if pose_folder.is_dir():
                    presets[pose_folder.name] = str(pose_folder)
        return presets
    
    def _load_reference_images(self):
        """加载预设参考图像"""
        ref_dir = self.project_root / "assets" / "halfbody_demo" / "refimag"
        images = {}
        if ref_dir.exists():
            for category in ref_dir.iterdir():
                if category.is_dir():
                    category_images = []
                    for img_file in category.glob("*.png"):
                        category_images.append((img_file.name, str(img_file)))
                    if category_images:
                        images[category.name] = category_images
        return images
    
    def preprocess_image(self, image_path, target_size=(512, 512)):
        """预处理用户上传的图像"""
        try:
            # 打开图像
            image = Image.open(image_path)
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 获取原始尺寸
            original_width, original_height = image.size
            
            # 计算缩放比例，保持长宽比
            aspect_ratio = original_width / original_height
            target_width, target_height = target_size
            
            if aspect_ratio > 1:  # 宽图
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:  # 高图或正方形
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # 缩放图像
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建目标尺寸的画布，背景为黑色
            canvas = Image.new('RGB', target_size, (0, 0, 0))
            
            # 计算粘贴位置（居中）
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # 粘贴图像
            canvas.paste(image, (paste_x, paste_y))
            
            # 保存处理后的图像
            processed_path = self.temp_dir / f"processed_{int(time.time())}.png"
            canvas.save(processed_path)
            
            return str(processed_path), f"✅ 图像处理完成\n原始尺寸: {original_width}x{original_height}\n处理后尺寸: {target_size[0]}x{target_size[1]}"
            
        except Exception as e:
            return None, f"❌ 图像处理失败: {str(e)}"
    
    def preprocess_audio(self, audio_path, target_sr=16000, max_duration=30):
        """预处理用户上传的音频"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=None)
            
            # 限制音频长度
            if len(audio) / sr > max_duration:
                audio = audio[:int(max_duration * sr)]
                duration_info = f"⚠️ 音频已截断至{max_duration}秒"
            else:
                duration_info = f"✅ 音频长度: {len(audio)/sr:.2f}秒"
            
            # 重采样到目标采样率
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # 归一化音频
            audio = librosa.util.normalize(audio)
            
            # 保存处理后的音频
            processed_path = self.temp_dir / f"processed_audio_{int(time.time())}.wav"
            sf.write(processed_path, audio, sr)
            
            return str(processed_path), f"{duration_info}\n采样率: {sr}Hz\n✅ 音频处理完成"
            
        except Exception as e:
            return None, f"❌ 音频处理失败: {str(e)}"
    
    def generate_video(self, reference_image, audio_file, pose_preset, use_custom_image, custom_image, use_custom_audio, custom_audio, progress=gr.Progress()):
        """生成数字人视频"""
        try:
            progress(0.1, "准备输入文件...")
            
            # 确定参考图像
            if use_custom_image and custom_image:
                progress(0.2, "处理自定义图像...")
                ref_img_path, img_msg = self.preprocess_image(custom_image)
                if not ref_img_path:
                    return None, f"❌ {img_msg}"
                status_msg = f"📸 使用自定义图像\n{img_msg}\n"
            else:
                ref_img_path = reference_image
                status_msg = f"📸 使用预设图像: {Path(reference_image).name}\n"
            
            # 确定音频文件
            if use_custom_audio and custom_audio:
                progress(0.3, "处理自定义音频...")
                audio_path, audio_msg = self.preprocess_audio(custom_audio)
                if not audio_path:
                    return None, f"❌ {audio_msg}"
                status_msg += f"🎵 使用自定义音频\n{audio_msg}\n"
            else:
                audio_path = audio_file
                status_msg += f"🎵 使用预设音频: {Path(audio_file).name}\n"
            
            # 确定姿势预设
            pose_path = self.pose_presets[pose_preset]
            status_msg += f"🤖 使用姿势预设: {pose_preset}\n"
            
            progress(0.4, "创建推理配置...")
            
            # 创建临时配置文件
            config_data = {
                'pose_dir': pose_path,
                'ref_image_path': ref_img_path,
                'audio_path': audio_path,
                'output_dir': str(self.output_dir),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            config_path = self.temp_dir / f"config_{int(time.time())}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            progress(0.5, "启动推理进程...")
            
            # 构建推理命令
            cmd = [
                sys.executable, 
                "infer.py", 
                f"--config={config_path}"
            ]
            
            # 运行推理
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # 实时读取输出
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line.strip())
                    # 更新进度
                    if "%" in line and "|" in line:
                        try:
                            percent_str = line.split('%')[0].split()[-1]
                            if percent_str.replace('.', '').isdigit():
                                percent = float(percent_str) / 100
                                progress(0.5 + percent * 0.4, f"生成视频... {percent_str}%")
                        except:
                            pass
            
            progress(0.95, "查找生成的视频...")
            
            # 查找生成的视频文件
            video_files = list(self.output_dir.rglob("*.mp4"))
            if video_files:
                # 找到最新的视频文件
                latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                
                progress(1.0, "视频生成完成!")
                
                return str(latest_video), f"✅ 视频生成成功!\n{status_msg}\n📁 输出路径: {latest_video}\n\n推理日志:\n" + "\n".join(output_lines[-20:])
            else:
                return None, f"❌ 未找到生成的视频文件\n{status_msg}\n\n推理日志:\n" + "\n".join(output_lines)
                
        except Exception as e:
            return None, f"❌ 生成失败: {str(e)}\n{status_msg}"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # CSS样式
        css = """
        .gradio-container {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
        }
        .title {
            text-align: center;
            color: #2E8B57;
            margin-bottom: 30px;
        }
        .section-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status-box {
            background-color: #f0f8f0;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        """
        
        with gr.Blocks(css=css, title="EchoMimic 数字人生成器") as interface:
            
            gr.HTML("""
            <h1 class="title">🎭 EchoMimic 数字人生成器</h1>
            <p style="text-align: center; color: #666; font-size: 18px;">
                创建属于你的个性化数字人视频 - 支持自定义图像和音频
            </p>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">📸 图像设置</div>')
                    
                    use_custom_image = gr.Checkbox(
                        label="使用自定义图像",
                        value=False,
                        info="上传你自己的照片作为数字人形象"
                    )
                    
                    with gr.Group(visible=True) as preset_image_group:
                        # 预设图像选择
                        ref_categories = list(self.reference_images.keys())
                        if ref_categories:
                            ref_category = gr.Dropdown(
                                choices=ref_categories,
                                label="图像类别",
                                value=ref_categories[0] if ref_categories else None
                            )
                            
                            ref_image = gr.Dropdown(
                                choices=[img[0] for img in self.reference_images.get(ref_categories[0], [])] if ref_categories else [],
                                label="选择预设图像",
                                value=self.reference_images[ref_categories[0]][0][0] if ref_categories and self.reference_images.get(ref_categories[0]) else None
                            )
                            
                            ref_preview = gr.Image(
                                label="预设图像预览",
                                interactive=False,
                                height=200
                            )
                    
                    with gr.Group(visible=False) as custom_image_group:
                        custom_image = gr.File(
                            label="上传你的照片",
                            file_types=[".jpg", ".jpeg", ".png", ".bmp"],
                            info="支持 JPG, PNG 等格式，建议使用清晰的正面照片"
                        )
                        
                        custom_preview = gr.Image(
                            label="自定义图像预览",
                            interactive=False,
                            height=200
                        )
                
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">🎵 音频设置</div>')
                    
                    use_custom_audio = gr.Checkbox(
                        label="使用自定义音频",
                        value=False,
                        info="上传你自己的语音文件"
                    )
                    
                    with gr.Group(visible=True) as preset_audio_group:
                        # 预设音频选择
                        audio_files = []
                        audio_dir = self.project_root / "assets" / "halfbody_demo" / "audio"
                        if audio_dir.exists():
                            for audio_file in audio_dir.rglob("*.wav"):
                                audio_files.append((audio_file.name, str(audio_file)))
                        
                        audio_file = gr.Dropdown(
                            choices=[audio[0] for audio in audio_files],
                            label="选择预设音频",
                            value=audio_files[0][0] if audio_files else None
                        )
                        
                        preset_audio_player = gr.Audio(
                            label="预设音频试听",
                            interactive=False
                        )
                    
                    with gr.Group(visible=False) as custom_audio_group:
                        custom_audio = gr.File(
                            label="上传你的音频",
                            file_types=[".wav", ".mp3", ".m4a", ".flac"],
                            info="支持 WAV, MP3 等格式，建议时长不超过30秒"
                        )
                        
                        custom_audio_player = gr.Audio(
                            label="自定义音频试听",
                            interactive=False
                        )
                
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">🤖 姿势设置</div>')
                    
                    pose_preset = gr.Dropdown(
                        choices=list(self.pose_presets.keys()),
                        label="选择姿势预设",
                        value=list(self.pose_presets.keys())[0] if self.pose_presets else None,
                        info="不同的姿势动作序列"
                    )
                    
                    gr.HTML('<div class="section-header">⚡ 生成控制</div>')
                    
                    generate_btn = gr.Button(
                        "🎬 生成数字人视频",
                        variant="primary",
                        size="lg"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">📺 输出结果</div>')
                    
                    output_video = gr.Video(
                        label="生成的数字人视频",
                        height=400
                    )
                    
                    status_output = gr.Textbox(
                        label="状态信息",
                        lines=10,
                        max_lines=15,
                        interactive=False,
                        elem_classes=["status-box"]
                    )
            
            # 事件处理函数
            def update_image_visibility(use_custom):
                return gr.Group.update(visible=not use_custom), gr.Group.update(visible=use_custom)
            
            def update_audio_visibility(use_custom):
                return gr.Group.update(visible=not use_custom), gr.Group.update(visible=use_custom)
            
            def update_ref_image_choices(category):
                if category in self.reference_images:
                    choices = [img[0] for img in self.reference_images[category]]
                    return gr.Dropdown.update(choices=choices, value=choices[0] if choices else None)
                return gr.Dropdown.update(choices=[], value=None)
            
            def update_ref_preview(category, image_name):
                if category in self.reference_images:
                    for img_name, img_path in self.reference_images[category]:
                        if img_name == image_name:
                            return img_path
                return None
            
            def update_custom_preview(file):
                if file:
                    return file.name
                return None
            
            def update_preset_audio(audio_name):
                audio_dir = self.project_root / "assets" / "halfbody_demo" / "audio"
                for audio_file in audio_dir.rglob("*.wav"):
                    if audio_file.name == audio_name:
                        return str(audio_file)
                return None
            
            def update_custom_audio_player(file):
                if file:
                    return file.name
                return None
            
            # 绑定事件
            use_custom_image.change(
                update_image_visibility,
                inputs=[use_custom_image],
                outputs=[preset_image_group, custom_image_group]
            )
            
            use_custom_audio.change(
                update_audio_visibility,
                inputs=[use_custom_audio],
                outputs=[preset_audio_group, custom_audio_group]
            )
            
            if ref_categories:
                ref_category.change(
                    update_ref_image_choices,
                    inputs=[ref_category],
                    outputs=[ref_image]
                )
                
                ref_image.change(
                    update_ref_preview,
                    inputs=[ref_category, ref_image],
                    outputs=[ref_preview]
                )
            
            custom_image.change(
                update_custom_preview,
                inputs=[custom_image],
                outputs=[custom_preview]
            )
            
            if audio_files:
                audio_file.change(
                    update_preset_audio,
                    inputs=[audio_file],
                    outputs=[preset_audio_player]
                )
            
            custom_audio.change(
                update_custom_audio_player,
                inputs=[custom_audio],
                outputs=[custom_audio_player]
            )
            
            # 生成按钮事件
            generate_btn.click(
                self.generate_video,
                inputs=[
                    # 需要传递实际路径，这里需要修改逻辑
                    gr.State(self.reference_images[ref_categories[0]][0][1] if ref_categories and self.reference_images.get(ref_categories[0]) else ""),
                    gr.State(audio_files[0][1] if audio_files else ""),
                    pose_preset,
                    use_custom_image,
                    custom_image,
                    use_custom_audio,
                    custom_audio
                ],
                outputs=[output_video, status_output]
            )
        
        return interface

# 启动应用
def main():
    app = EchoMimicWebApp()
    interface = app.create_interface()
    
    # 启动服务器
    interface.launch(
        server_name="127.0.0.1",  # 本地访问
        server_port=7860,        # 端口
        share=False,             # 不创建公共链接
        debug=True,              # 调试模式
        show_error=True,         # 显示错误
        inbrowser=True           # 自动打开浏览器
    )

if __name__ == "__main__":
    main()