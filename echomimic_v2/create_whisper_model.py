import whisper
import torch

# 加载whisper模型
model = whisper.load_model('tiny')

# 创建兼容的checkpoint格式
checkpoint = {
    'dims': {
        'n_mels': model.dims.n_mels,
        'n_vocab': model.dims.n_vocab,
        'n_audio_ctx': model.dims.n_audio_ctx,
        'n_audio_state': model.dims.n_audio_state,
        'n_audio_head': model.dims.n_audio_head,
        'n_audio_layer': model.dims.n_audio_layer,
        'n_text_ctx': model.dims.n_text_ctx,
        'n_text_state': model.dims.n_text_state,
        'n_text_head': model.dims.n_text_head,
        'n_text_layer': model.dims.n_text_layer
    }
}
# 添加模型状态
checkpoint.update(model.state_dict())

# 保存
torch.save(checkpoint, 'pretrained_weights/audio_processor/tiny.pt')
print('Compatible Whisper model created successfully')