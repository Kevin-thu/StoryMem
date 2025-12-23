from hpsv3 import HPSv3RewardInferencer
import os
import time
import glob
import io
import math
import cv2
import decord
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from typing import Tuple, List
# from utils.image_quality_score import is_low_quality

import logging
logger = logging.getLogger()

IMAGE_FACTOR = 28
# min tokens per image
MIN_TOKENS = 4
# max tokens per image
MAX_TOKENS = 20480
MIN_PIXELS = MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 4 * 28 * 28 = 3,136
MAX_PIXELS = MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 20480 * 28 * 28 = 16,056,320
MAX_RATIO = 200
# min tokens per video frame
VIDEO_MIN_TOKENS = 48
# max tokens per video frame
VIDEO_MAX_TOKENS = 768
# min pixels per video frame
VIDEO_MIN_PIXELS = VIDEO_MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 48 * 28 * 28 = 25,088
# max pixels per video frame
VIDEO_MAX_PIXELS = VIDEO_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 768 * 28 * 28 = 602,112
# max total pixels per video
VIDEO_TOTAL_PIXELS = 65536 * IMAGE_FACTOR * IMAGE_FACTOR # 65,536 * 28 * 28 = 51,380,224
# initila default min frame similarity for selecting keyframes
MIN_FRAME_SIMILARITY = 0.9
# max keyframe number
MAX_KEYFRAME_NUM = 3
# alpha for adaptive threshold
ADAPTIVE_ALPHA = 0.01
HPSV3_QUALITY_THRESHOLD = 3.0

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
        height: int, width: int,
        factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return max(h_bar, factor), max(w_bar, factor)

def get_frame_sim(frame1, frame2,
                  patch_size: int=28,
                  threshold: float = 0.7,
                  epsilon: float=1e-8):
    assert frame1.dim() == 3 and frame2.dim() == 3, "Input must be a 3-D tensor [C, H, W]"
    
    def to_numpy_cvt(tensor):
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            tensor = (tensor).astype(np.uint8)
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2HSV)

    frame1_hsv = to_numpy_cvt(frame1)
    frame2_hsv = to_numpy_cvt(frame2)

    frame1_tensor = torch.from_numpy(frame1_hsv).permute(2, 0, 1).to(frame1.device).float()
    frame2_tensor = torch.from_numpy(frame2_hsv).permute(2, 0, 1).to(frame2.device).float()

    patch1 = rearrange(
        frame1_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(
        frame2_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    
    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)
    
    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)
    
    similar = torch.ones_like(cos_sim)
    
    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > threshold).float()

    return similar[non_zero_mask].float().mean().item()

# 单例缓存
class _CLIPCtx:
    model = None
    device = None
    dtype = torch.float32

def _get_clip_model(device="cpu", use_half=False):
    if _CLIPCtx.model is not None:
        return _CLIPCtx.model, _CLIPCtx.device, _CLIPCtx.dtype

    dev = device # or ("cuda:0" if torch.cuda.is_available() else "cpu")
    import clip  # 来自 openai/CLIP

    model, _ = clip.load("ViT-B/32", device=dev, jit=False)  # 224 输入
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dt = torch.float16 if (use_half and dev != "cpu") else torch.float32
    if dt == torch.float16:
        model = model.half()

    _CLIPCtx.model, _CLIPCtx.device, _CLIPCtx.dtype = model, dev, dt
    return model, dev, dt

# class _HPSv3:
#     model = None
#     device = None

# def _get_quality_model(device="cuda"):
#     """
#     加载 HPSv3 模型，优先 CUDA。
#     """
#     if _HPSv3.model is not None:
#         return _HPSv3.model, _HPSv3.device
#     dev = device # or ("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = HPSv3RewardInferencer(device=device)
#     _HPSv3.model, _HPSv3.device = model, dev
#     return model, dev

def is_low_quality(frame, quality_model, threshold=HPSV3_QUALITY_THRESHOLD):
    frame = frame.permute(1, 2, 0).numpy().astype(np.uint8).clip(0, 255)
    rewards = quality_model.reward(image_paths=[Image.fromarray(frame)], prompts=[""])
    score = rewards[0][0].item()
    return score < threshold

def _ensure_rgb01(chw: torch.Tensor):
    assert chw.dim() == 3 and chw.shape[0] in (1,3), f"expect (C,H,W), got {tuple(chw.shape)}"
    x = chw
    if not torch.is_floating_point(x):
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    if x.shape[0] == 1:
        x = x.repeat(3,1,1)
    return x.clamp(0.0, 1.0)

def _clip_preprocess_tensor(x_chw: torch.Tensor, size=224):
    x = x_chw.unsqueeze(0)  # (1,3,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    return x

@torch.no_grad()
def get_frame_sim_clip(frame1: torch.Tensor, frame2: torch.Tensor, device=None, use_half=False) -> float:
    model, dev, dt = _get_clip_model()

    f1 = _ensure_rgb01(frame1).to(dev)
    f2 = _ensure_rgb01(frame2).to(dev)

    x1 = _clip_preprocess_tensor(f1).to(dev, dtype=dt)
    x2 = _clip_preprocess_tensor(f2).to(dev, dtype=dt)

    z1 = model.encode_image(x1)
    z2 = model.encode_image(x2)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cos = (z1 * z2).sum(dim=-1)  # (1,)
    return cos.item()

def extract_keyframe_indices(frames, quality_model, threshold = MIN_FRAME_SIMILARITY, frame_sim_func = get_frame_sim_clip):
    assert frames.dim() == 4, "Input must be a 4-D tensor [N, C, H, W]"

    num_frames, _, height, width = frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=256 * IMAGE_FACTOR * IMAGE_FACTOR,
    )

    resized_frames = nn.functional.interpolate(
        frames,
        [resized_height, resized_width],
        mode="bilinear",
        antialias=True,
    ).float()

    first_keyframe_indice = 0
    while is_low_quality(resized_frames[first_keyframe_indice], quality_model):
        first_keyframe_indice += 1
        if first_keyframe_indice >= num_frames:
            return []
    keyframe_indices = [first_keyframe_indice]
    last_keyframe = resized_frames[first_keyframe_indice]
    for i in range(2, resized_frames.size(0)):
        current_frame = resized_frames[i]
        sim = frame_sim_func(last_keyframe, current_frame)
        # print(f"? sim={sim}")

        if sim < threshold and not is_low_quality(current_frame, quality_model):
            keyframe_indices.append(i)
            last_keyframe = current_frame
    
    return keyframe_indices

def read_video(
        video: str | bytes,
) -> Tuple[torch.Tensor, List[int]]:
    if isinstance(video, bytes):
        fp = io.BytesIO(video)
        vr = decord.VideoReader(fp)
    else:
        vr = decord.VideoReader(video)
    nframes, video_fps = len(vr), vr.get_avg_fps()
    timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])
    
    indices = torch.linspace(0, nframes - 1, nframes).round().long()
    frames = vr.get_batch(indices.tolist()).asnumpy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # T, C, H, W
    timestamps = timestamps[indices]
    return frames, timestamps

def save_keyframes(video_path, frame_sim_thereshold = MIN_FRAME_SIMILARITY, frame_sim_func = get_frame_sim_clip, memory_cmp = True):
    st = time.time()
    frames, timestamps = read_video(video_path)
    quality_model = HPSv3RewardInferencer(device="cuda")
    while True:
        keyframe_indices = extract_keyframe_indices(frames, quality_model, frame_sim_thereshold, frame_sim_func)
        # print(f"? {keyframe_indices}")
        if len(keyframe_indices) > MAX_KEYFRAME_NUM:
            frame_sim_thereshold -= ADAPTIVE_ALPHA
        else:
            break
    # keyframe_indices = [1, 40, 80]
    logger.info(f"Read video: {video_path=}, {keyframe_indices=}, time={time.time() - st:.3f}s")
    keyframes = frames[keyframe_indices]
    if memory_cmp:
        memory_paths = sorted(glob.glob(os.path.join(os.path.dirname(video_path), "*keyframe*.jpg")))
        memory_bank = []
        for p in memory_paths:
            memory_frame = cv2.imread(p)
            memory_frame = cv2.cvtColor(memory_frame, cv2.COLOR_BGR2RGB)
            memory_frame = torch.tensor(memory_frame).permute(2, 0, 1)
            memory_bank.append(memory_frame)
    for i, keyframe in enumerate(keyframes):
        if memory_cmp:
            pass_flag = False
            for memory in memory_bank:
                sim = frame_sim_func(keyframe, memory)
                if sim > MIN_FRAME_SIMILARITY:
                    pass_flag = True
                    break
            if pass_flag:
                continue
        keyframe = keyframe.permute(1, 2, 0).numpy()
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
        cv2.imwrite(video_path.replace(".mp4", f"_keyframe{i}.jpg"), keyframe)
    last_frame = frames[-1]
    last_frame = last_frame.permute(1, 2, 0).numpy()
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(os.path.dirname(video_path), "last_frame.jpg"), last_frame)

    last_frames = frames[-5:]
    motion_frames_path = os.path.join(os.path.dirname(video_path), "motion_frames.mp4")
    _, c, h, w = last_frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 5
    writer = cv2.VideoWriter(motion_frames_path, fourcc, fps, (w, h))
    for frame in last_frames:
        frame = frame.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()