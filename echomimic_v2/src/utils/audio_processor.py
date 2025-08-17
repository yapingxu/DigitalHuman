# -*- coding: utf-8 -*-
"""
Audio Processor for EchoMimic
Simple audio processing utilities
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, self.sample_rate
    
    def process_audio(self, audio_path):
        """Process audio and return features"""
        waveform, sr = self.load_audio(audio_path)
        return {
            'waveform': waveform,
            'sample_rate': sr,
            'duration': waveform.shape[1] / sr
        }