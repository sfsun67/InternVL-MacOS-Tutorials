# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

'''
        # from apex.normalization import FusedRMSNorm 
是从 NVIDIA 的 Apex 库中导入 FusedRMSNorm 模块。Apex 是 NVIDIA 提供的一个开源库，旨在帮助加速深度学习模型的训练，特别是在混合精度训练（Mixed Precision Training）和分布式训练方面。

FusedRMSNorm 是一种融合的均方根归一化（RMSNorm）实现，利用 CUDA 的加速能力，提高了计算性能和数值稳定性。在深度学习模型中，RMSNorm 是一种用于标准化神经网络层输出的技术，有助于加速模型收敛并提高性能。

需要注意的是，使用 FusedRMSNorm 需要安装带有 CUDA 扩展的 Apex。如果未安装 Apex，可能会出现类似 “Nvidia APEX 未安装，使用 PyTorch 的 LayerNorm” 的警告信息。此时，系统会退回使用 PyTorch 自带的 LayerNorm，但性能可能不及 Apex 的优化实现。

如果您在安装 Apex 时遇到问题，可以参考以下资源获取帮助：
	•	Apex GitHub 仓库
	•	Apex 安装和使用教程
	•	解决 Apex 安装问题的经验分享

通过正确安装和配置 Apex，您可以充分利用 FusedRMSNorm 等优化模块，提升深度学习模型的训练效率。
'''

import transformers


def replace_llama_rmsnorm_with_fused_rmsnorm():
    try:
        from functools import partial

        from apex.normalization import FusedRMSNorm
        LlamaRMSNorm = partial(FusedRMSNorm, eps=1e-6)   # noqa
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        print('Discovered apex.normalization.FusedRMSNorm - will use it instead of LlamaRMSNorm')
    except ImportError:
        # using the normal LlamaRMSNorm
        pass
    except Exception:
        print('discovered apex but it failed to load, falling back to LlamaRMSNorm')
        pass
