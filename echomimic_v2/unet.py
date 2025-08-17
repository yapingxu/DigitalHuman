# -*- coding: utf-8 -*-
"""
UNet 模块导入桥梁
为了保持兼容性，从具体的 UNet 实现文件中导入所需的类
"""

try:
    # 尝试从 unet_3d.py 导入 UNet3DConditionModel
    from .unet_3d import UNet3DConditionModel
except ImportError:
    try:
        # 如果失败，尝试从 unet_3d_emo.py 导入
        from .unet_3d_emo import UNet3DConditionModel
    except ImportError:
        # 最后尝试从 unet_2d_condition.py 导入
        from .unet_2d_condition import UNet3DConditionModel

# 导出所有需要的类
__all__ = ['UNet3DConditionModel']
