import whisper
import os
import shutil

# 确保whisper模型被下载
model = whisper.load_model('tiny')

# 查找缓存目录
whisper_cache = os.path.join(os.path.expanduser('~'), '.cache', 'whisper')

print('Looking for whisper cache in:', whisper_cache)
if os.path.exists(whisper_cache):
    files = os.listdir(whisper_cache)
    print('Found files:', files)
    for f in files:
        if 'tiny' in f and f.endswith('.pt'):
            src = os.path.join(whisper_cache, f)
            shutil.copy2(src, 'pretrained_weights/audio_processor/tiny.pt')
            print(f'Copied {f} to tiny.pt')
            break
else:
    print('Cache directory not found')