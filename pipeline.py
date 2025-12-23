import os
import gc
import sys
import glob
import json5
import argparse
import subprocess
import logging
import torch
import torch.distributed as dist
import multiprocessing as mp
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool
from extract_keyframes import save_keyframes


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--story_script_path", type=str, default="./story/neo.json")
    parser.add_argument("--t2v_model_path", type=str, default="/path/to/Wan2.2-T2V-A14B")
    parser.add_argument("--i2v_model_path", type=str, default="/path/to/Wan2.2-I2V-A14B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--max_memory_size", type=int, default=8)
    parser.add_argument("--input_dir", type=str, default="./input")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Whether to convert model paramerters dtype.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample.")
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=3.5, help="Classifier free guidance scale.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1")
    parser.add_argument("--offload_model", action="store_true", help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument("--t2v_first_shot", action="store_true", help="Whether to generate the first shot with T2V model.")
    parser.add_argument("--m2v_first_shot", action="store_true", help="Whether to generate the first shot with M2V model.")
    parser.add_argument("--mi2v", action="store_true", help="Whether to start from last frame of last video shot with MI2V")
    parser.add_argument("--mm2v", action="store_true", help="Whether to start from last frame of last video shot with MM2V")
    parser.add_argument("--fix", type=int, default=3, help="Whether to fix the first n keyframes.")
    parser.add_argument("--finetune_checkpoint_dir", type=str, default=None, help="The path to the finetune checkpoint.")
    parser.add_argument("--lora_weight_path", type=str, default=None, help="The path to the LoRA weight.")
    parser.add_argument("--lora_rank", type=int, default=None, help="The rank of LoRA weight.")
    args = parser.parse_args()
    return args

def _init_logging(rank, log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout),
                logging.FileHandler(log_file, mode='a', encoding='utf-8')
            ])
    else:
        logging.basicConfig(
            level=logging.ERROR,
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )

def main(args):
    ###### Init ######
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank, args.log_file)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if dist.is_initialized():
        base_seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.seed = base_seed[0]

    story_script = json5.load(open(args.story_script_path, "r", encoding="utf-8"))
    os.makedirs(args.output_dir, exist_ok=True)

    ###### Generate first-shot videos ######
    if args.t2v_first_shot:
        t2v_config = WAN_CONFIGS["t2v-A14B"]

        logging.info("Loading T2V model...")
        t2v_model = wan.WanT2V(
            config=t2v_config,
            checkpoint_dir=args.t2v_model_path,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        prompt = story_script["scenes"][0]["video_prompts"][0]
        logging.info(f"Generating Scene 1 / Shot 1: {prompt}")
        # if args.use_prompt_extend:
        #     logging.info("Extending prompt ...")
        #     if rank == 0:
        #         prompt_output = prompt_expander(
        #             prompt,
        #             tar_lang="en",
        #             seed=args.base_seed)
        #         if prompt_output.status == False:
        #             logging.info(
        #                 f"Extending prompt failed: {prompt_output.message}")
        #             logging.info("Falling back to original prompt.")
        #             input_prompt = args.prompt
        #         else:
        #             input_prompt = prompt_output.prompt
        #         input_prompt = [input_prompt]
        #     else:
        #         input_prompt = [None]
        #     if dist.is_initialized():
        #         dist.broadcast_object_list(input_prompt, src=0)
        #     prompt = input_prompt[0]
        #     logging.info(f"Extended prompt: {prompt}")

        video = t2v_model.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=t2v_config.frame_num,
            shift=t2v_config.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=t2v_config.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.seed,
            offload_model=args.offload_model
        )

        if rank == 0:
            save_video(
                tensor=video[None],
                save_file=f"{args.output_dir}/01_01.mp4",
                fps=t2v_config.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            save_keyframes(f"{args.output_dir}/01_01.mp4")
            del video
            torch.cuda.empty_cache()

        del t2v_model

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    ###### Generate next-shot videos ######
    m2v_config = WAN_CONFIGS["m2v-A14B"]
    if args.lora_weight_path is not None:
        m2v_config.low_noise_lora.weight = os.path.join(args.lora_weight_path, "backbone_low_noise.pth")
        m2v_config.high_noise_lora.weight = os.path.join(args.lora_weight_path, "backbone_high_noise.pth")
    if args.lora_rank is not None:
        m2v_config.low_noise_lora.r = m2v_config.low_noise_lora.lora_alpha = args.lora_rank
        m2v_config.high_noise_lora.r = m2v_config.high_noise_lora.lora_alpha = args.lora_rank

    logging.info("Loading M2V model...")
    m2v_model = wan.WanM2V(
        config=m2v_config,
        checkpoint_dir=args.i2v_model_path,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        finetune_checkpoint_dir=args.finetune_checkpoint_dir,
    )

    for scene in story_script["scenes"]:
        scene_num = scene["scene_num"]
        # os.makedirs(f"{args.output_dir}/scene_{scene_num}", exist_ok=True)

        for i, prompt in enumerate(scene["video_prompts"]):
            shot_num = i + 1
            if (not args.m2v_first_shot) and scene_num == 1 and shot_num == 1:
                continue
            logging.info(f"Generating Scene {scene_num} / Shot {shot_num}: {prompt}")

            # if args.use_prompt_extend:
            #     logging.info("Extending prompt ...")
            #     if rank == 0:
            #         prompt_output = prompt_expander(
            #             args.prompt,
            #             image=img,
            #             tar_lang=args.prompt_extend_target_lang,
            #             seed=args.base_seed)
            #         if prompt_output.status == False:
            #             logging.info(
            #                 f"Extending prompt failed: {prompt_output.message}")
            #             logging.info("Falling back to original prompt.")
            #             input_prompt = args.prompt
            #         else:
            #             input_prompt = prompt_output.prompt
            #         input_prompt = [input_prompt]
            #     else:
            #         input_prompt = [None]
            #     if dist.is_initialized():
            #         dist.broadcast_object_list(input_prompt, src=0)
            #     args.prompt = input_prompt[0]
            #     logging.info(f"Extended prompt: {args.prompt}")

            memory_bank = sorted(glob.glob(f"{args.output_dir}/*keyframe*.jpg"))
            # TODO: currently use sliding window strategy
            # if args.fix is None:
            if len(memory_bank) > args.max_memory_size:
                memory_bank = memory_bank[:args.fix] + memory_bank[-(args.max_memory_size-args.fix):]
            # else:
            #     memory_bank = memory_bank[:args.fix]
            if args.mi2v and not scene["cut"][i]:
                first_frame_file = f"{args.output_dir}/last_frame.jpg"
            else:
                first_frame_file = None
            if args.mm2v and not scene["cut"][i]:
                motion_frames_file = f"{args.output_dir}/motion_frames.mp4"
            else:
                motion_frames_file = None
            video = m2v_model.generate(
                prompt,
                memory_bank,
                first_frame_file=first_frame_file,
                motion_frames_file=motion_frames_file,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=m2v_config.frame_num,
                shift=m2v_config.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=m2v_config.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.seed+i,
                offload_model=args.offload_model
            )

            if rank == 0:
                if first_frame_file is not None:
                    video = video[:, 1:]
                elif motion_frames_file is not None:
                    video = video[:, 5:]
                save_video(
                    tensor=video[None],
                    save_file=f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4",
                    fps=m2v_config.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                save_keyframes(f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4")
                del video
                torch.cuda.empty_cache()

            if dist.is_initialized():
                dist.barrier()
    
    videos = sorted(glob.glob(f"{args.output_dir}/*.mp4"))
    list_path = os.path.join(args.output_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for v in videos:
            f.write(f"file '{os.path.abspath(v)}'\n")
    out = os.path.join(args.output_dir, f"{os.path.basename(args.output_dir)}.mp4")
    ret = subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-y", out]
    )
    if ret.returncode != 0:
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-r", "30", "-y", out
        ], check=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")

if __name__ == "__main__":
    args = _parse_args()
    main(args)