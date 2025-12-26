import gc
import logging
import math
import os
import random
import sys
import types
import copy
import av
from PIL import Image
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from decord import VideoReader, cpu
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import logging
logger = logging.getLogger()

class WanM2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        finetune_checkpoint_dir=None
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        if finetune_checkpoint_dir is not None:
            finetune_low_noise_model_path = os.path.join(finetune_checkpoint_dir, "backbone_low_noise.pth")
            logger.info(f"Loading finetune model from {finetune_low_noise_model_path}")
            state_dict = torch.load(finetune_low_noise_model_path, map_location="cpu", mmap=True)
            self.low_noise_model.load_state_dict(state_dict, strict=True)
        if "low_noise_lora" in config and config.low_noise_lora.enabled:
            self.low_noise_model = self._load_lora(
                self.low_noise_model, config.low_noise_lora)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        if finetune_checkpoint_dir is not None:
            finetune_high_noise_model_path = os.path.join(finetune_high_noise_model_path, "backbone_high_noise.pth")
            logger.info(f"Loading finetune model from {finetune_high_noise_model_path}")
            state_dict = torch.load(finetune_high_noise_model_path, map_location="cpu", mmap=True)
            self.high_noise_model.load_state_dict(state_dict, strict=True)
        if "high_noise_lora" in config and config.high_noise_lora.enabled:
            self.high_noise_model = self._load_lora(
                self.high_noise_model, config.high_noise_lora)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _load_lora(self, model, lora_config):
        """
        Load LoRA weights.
        """
        lora_cfg = copy.deepcopy(lora_config)
        lora_cfg.pop("enabled")
        lora_weight = lora_cfg.pop("weight", None)

        lora_cfg = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_cfg)

        if lora_weight is not None:
            logger.info(f"Loading LoRA from {lora_weight}")

            ext = os.path.splitext(lora_weight)[1]

            if ext == ".safetensors":
                state_dict = load_file(lora_weight, device="cpu")
            else:
                state_dict = torch.load(
                    lora_weight,
                    map_location="cpu",
                    mmap=True
                )

            msg = set_peft_model_state_dict(model, state_dict)
            # logger.info(f"Missing keys: {msg.missing_keys}")
        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def _vae_decode(self, z, with_memory = False, memory_size = 0):
        """
        z: latent shape [lat_C, lat_T, lat_H, lat_W]
        """
        if not with_memory:
            self.vae.decode(z.unsqueeze(0)).float().clamp_(-1, 1).squeeze(0)
        memory_latent, video_latent = z[:, :memory_size], z[:, memory_size:]
        memory_latent = memory_latent.permute(1, 0, 2, 3).unsqueeze(2)
        memory = self.vae.decode(memory_latent).float().clamp_(-1, 1).squeeze(2) # M, C, H, W
        video = self.vae.decode(video_latent.unsqueeze(0)).float().clamp_(-1, 1).squeeze(0)
        return video, memory

    @torch.no_grad()
    def generate(self,
                 input_prompt,
                 memory_files,
                 first_frame_file=None,
                 motion_frames_file=None,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 fix_mem=None,
                 seed=-1,
                 offload_model=True,
                 return_memory=False):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            memory_files (List[str]):
                Input memory files.
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        assert first_frame_file is None or motion_frames_file is None
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        memory = []
        for filename in memory_files:
            if filename.endswith(".mp4"):
                container = av.open(filename)
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                frames = []
                for frame in container.decode(stream):
                    img = frame.to_image().convert("RGB")
                    frames.append(img)
                total = len(frames)
                if total == 0:
                    raise RuntimeError(f"No video frames decoded from {filename}")
                idxs = [0, total - 1]
                mids = random.sample(list(range(1, total - 1)), 2)
                idxs.extend(mids)
                idxs = sorted(idxs)
                for i in idxs:
                    img = frames[i]
                    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
                    memory.append(img)
                container.close()            
            else:
                img = Image.open(filename).convert("RGB")
                img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
                memory.append(img)

        memory_size = len(memory)
        F = frame_num
        h, w = memory[0].shape[1:]

        if first_frame_file is not None:
            first_frame = Image.open(first_frame_file).convert("RGB")
            first_frame = TF.to_tensor(first_frame).sub_(0.5).div_(0.5).to(self.device)
            h, w = first_frame.shape[1:]

        if motion_frames_file is not None:
            vr = VideoReader(motion_frames_file, ctx=cpu())
            assert len(vr) == 5
            frames = []
            for i in range(5):
                frame = vr[i].asnumpy()
                frame = Image.fromarray(frame).convert("RGB")
                frame = TF.to_tensor(frame).sub_(0.5).div_(0.5).to(self.device)
                frames.append(frame)
            motion_frames = torch.stack(frames, dim=0) # [T, 3, H, W]
            h, w = motion_frames.shape[-2:]

        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio)) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        lat_w = round(np.sqrt(max_area / aspect_ratio)) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        lat_t = (F - 1) // self.vae_stride[0] + 1 + memory_size
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = lat_t * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_t,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F + memory_size, lat_h, lat_w, device=self.device)
        if fix_mem is not None and memory_size > fix_mem:
            msk[:, fix_mem:memory_size] = 2
        msk[:, memory_size:] = 0
        if first_frame_file is not None:
            msk[:, memory_size] = 1
        if motion_frames_file is not None:
            msk[:, memory_size:memory_size + 5] = 1
        msk = torch.concat([
            torch.repeat_interleave(msk[:, :memory_size + 1], repeats=self.vae_stride[0], dim=1), msk[:, memory_size + 1:]
        ], dim=1)  # [1, 1, 1, 1] [0, 0, 0, 0] ..... [0, 0, 0, 0] Since the frist frame is encoded saparately in VAE
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        memory = [torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic') for img in memory]
        memory = torch.cat(memory, dim=0).to(self.device)
        memory = self.vae.encode(memory.unsqueeze(2)).float().squeeze(2).permute(1, 0, 2, 3) # M, C, 1, H, W -> C, M, H, W
        if first_frame_file is not None:
            u = self.vae.encode(
                torch.concat([
                    torch.nn.functional.interpolate(first_frame[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F-1, h, w)
                ], dim=1).unsqueeze(0).to(self.device)
            ).float().squeeze(0)
        if motion_frames_file is not None:
            u = self.vae.encode(
                torch.concat([
                    motion_frames.transpose(0, 1).to(self.device),
                    torch.zeros(3, F-5, h, w).to(self.device)
                ], dim=1).unsqueeze(0)
            ).float().squeeze(0)
        else:
            u = torch.zeros(3, F, h, w).to(self.device)
            u = self.vae.encode(u.unsqueeze(0)).float().squeeze(0)
        y = torch.cat([memory, u], dim=1)
        y = torch.cat([msk, y], dim=0)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = latent
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                video, memory = self._vae_decode(x0, with_memory=True, memory_size=memory_size)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if return_memory:
            if self.rank == 0:
                return video, memory
            else:
                return None, None
        else:
            return video if self.rank == 0 else None
