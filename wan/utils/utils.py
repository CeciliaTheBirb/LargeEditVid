# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp

import imageio
import torch
import torchvision

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


import os.path as osp
import torch
import torchvision
import imageio
import numpy as np
import logging
from typing import Tuple, Union

def cache_video(
    tensor: torch.Tensor,
    save_file: Union[str, None] = None,
    fps: int = 30,
    suffix: str = ".mp4",
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    retry: int = 5,
):
    """
    Save a 5-D tensor (B, C, T, H, W) as an mp4 file laid out in grids.

    If `save_file` is None, the video is cached to /tmp/[random_name].mp4.

    All frames are clamped to `value_range`, re-scaled to 0-255,
    and converted to uint8 before writing, preventing float→byte
    casting errors.
    """
    # 1) 生成临时保存路径
    if save_file is None:
        cache_file = osp.join("/tmp", rand_name(suffix=suffix))
    else:
        cache_file = save_file

    lo, hi = value_range            # 解包区间

    # 2) 多次尝试，防止写文件偶尔失败
    last_error = None
    for _ in range(retry):
        try:
            # -------- 预处理 -------- #
            # a) 保证浮点 & 连续
            tensor = tensor.float().contiguous()

            # b) 限幅到 [lo, hi]
            tensor = tensor.clamp(min=lo, max=hi)

            # c) 将每一帧做 make_grid → 组成 (T, H, W, C)
            frames = [
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=(lo, hi)
                ).permute(1, 2, 0)          # CHW → HWC
                for u in tensor.unbind(dim=2)  # dim=2 是时间维
            ]
            frames = torch.stack(frames, dim=0)  # (T, H, W, C)

            # d) [0,1] → [0,255] 并转 uint8
            frames = (frames * 255.0).round().to(dtype=torch.uint8, copy=False)

            # -------- 写视频 -------- #
            writer = imageio.get_writer(
                cache_file, fps=fps, codec="libx264", quality=8
            )

            for frame in frames.cpu().numpy():
                # imageio 需要 uint8 np.ndarray (H, W, C)
                writer.append_data(frame)

            writer.close()
            return cache_file

        except Exception as e:
            last_error = e
            logging.warning("cache_video attempt failed: %s", e, exc_info=False)
            continue

    # 所有重试均失败
    print(f"cache_video failed, error: {last_error}", flush=True)
    return None

def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
