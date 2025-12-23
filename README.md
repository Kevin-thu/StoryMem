<p align="center">
  <h2 align="center"><i>StoryMem</i>: Multi-shot Long Video Storytelling with Memory</h1>
  <!-- <h3 align="center">ICCV 2025</h3> -->
  <p align="center">
                <span class="author-block">
                <a href="https://kevin-thu.github.io/homepage/" target="_blank">Kaiwen Zhang</a>,
              </span>
              <span class="author-block">
                <a href="https://liming-jiang.com/" target="_blank">Liming Jiang</a><sup>✝</sup>,
              </span>
              <span class="author-block">
                <a href="https://angtianwang.github.io/" target="_blank">Angtian Wang</a>,
              </span>
              <span class="author-block">
                <a href="https://www.jacobzhiyuanfang.me/" target="_blank">Jacob Zhiyuan Fang</a>,
              </span><br>
              <span class="author-block">
                <a href="https://tiancheng-zhi.github.io/" target="_blank">Tiancheng Zhi</a>,
              </span>
              <span class="author-block">
                <a href="https://scholar.google.com/citations?user=0TIYjPAAAAAJ&hl=en" target="_blank">Qing Yan</a>,
              </span>
              <span class="author-block">
                <a href="https://scholar.google.com/citations?user=VeTCSyEAAAAJ" target="_blank">Hao Kang</a>,
              </span>
              <span class="author-block">
                <a href="https://scholar.google.com/citations?user=mFC0wp8AAAAJ&hl=en" target="_blank">Xin Lu</a>,
              </span>
              <span class="author-block">
                <a href="https://xingangpan.github.io/" target="_blank">Xingang Pan</a><sup>§</sup>
  </p>

  <p align="center">
    <sep>✝</sep>Project Lead &nbsp;
    <sep>§</sep>Corresponding Author
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2512.19539"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2512.19539-b31b1b.svg"></a>
    <a href="https://kevin-thu.github.io/StoryMem/"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
  <a href="https://huggingface.co/Kevin-thu/StoryMem"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange"></a>
    <!-- <a href="https://twitter.com/sze68zkw"><img alt='Twitter' src="https://img.shields.io/twitter/follow/sze68zkw?label=%40KaiwenZhang"></a> -->
  </p>

  <div align="center">
    <img src="./assets/teaser.png", width="800">
    <p align="left">Given a story script with per-shot text descriptions, StoryMem generates appealing minute-long, multi-shot narrative videos with highly coherent characters and cinematic visual quality. This is achieved through shot-by-shot generation using a memory-conditioned single-shot video diffusion model. <b>See our <a href="https://kevin-thu.github.io/StoryMem/">🌐 Project Page</a> for more details and video results.</b>
</p>
    
  </div>
</p>


## 🚀 Getting Started
### Installation
```bash
git clone --single-branch --branch main git@github.com:Kevin-thu/StoryMem.git
cd StoryMem
conda create -n storymem python=3.11
conda activate storymem
pip install -r requirements.txt
pip install flash_attn
```
<!-- If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` at last. -->
<!-- To run the code with CUDA properly, you can comment out `torch` and `torchvision` in `requirement.txt`, and install the appropriate version of `torch>=2.1.0+cu121` and `torchvision>=0.16.0+cu121` according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/). -->

### Model Download
Download Wan2.2 Base Model and *StoryMem* M2V LoRA from Huggingface:
| Models | Download Links | Description |
|---------------|---------------|---------------|
| Wan2.2 T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model |
| Wan2.2 I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model |
| *StoryMem* Wan2.2 M2V-A14B | 🤗 [Huggingface](https://huggingface.co/Kevin-thu/StoryMem)    | Memory-to-Video Fine-tuned LoRA |

You can easily download models using `huggingface-cli`:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./models/Wan2.2-I2V-A14B
huggingface-cli download Kevin-thu/StoryMem --local-dir ./models/StoryMem
```

There are two models provided:
- `StoryMem/Wan2.2-MI2V-A14B`: Support M2V and MI2V (memory + first-frame image conditioning)
- `StoryMem/Wan2.2-MM2V-A14B`: Support M2V and MM2V (memory + first 5 motion frames conditioning)

### Run the Code
You can run an example using the following command:
```bash
bash run_example.sh
```
This script first uses the T2V model to generate the first shot as the initial memory. It then uses our M2V model to generate the remaining shots shot by shot, automatically extracting keyframes and updating the memory after each shot.

**Key arguments**:
- `story_script_path`: Path to the story script JSON file.
- `output_dir`: Directory to save the generated videos. Default is `./results`.
- `t2v_model_path`: Path to the T2V model. Default is `./models/Wan2.2-T2V-A14B`.
- `i2v_model_path`: Path to the I2V model. Default is `./models/Wan2.2-I2V-A14B`.
- `lora_weight_path`: Path to the M2V LoRA weights. Default is `./models/StoryMem`.
- `seed`: Random seed. Default is `0`.
- `size`: Output video resolution. Default is `832*480`.
- `max_memory_size`: Maximum number of shots to keep in memory. Default is `10`.
- `t2v_first_shot`: Use T2V to generate the first shot as the initial memory.
- `m2v_first_shot`: Use M2V to generate the first shot (for the MR2V setting, where reference images are provided as initial memory). Reference images should be placed in `output_dir` as `00_00_keyframe0.jpg`, ..., `00_00_keyframeN.jpg`.
- `mi2v`: Enable MI2V (memory + first-frame image conditioning) to connect adjacent shots when `scene_cut` is `False`.
- `mm2v`: Enable MM2V (memory + first 5 motion frames conditioning) to connect adjacent shots when `scene_cut` is `False`.

### ST-Bench
To support evaluating multi-shot long video storytelling, we establish and release *ST-Bench*, which can be found in the subfloder `./story`.
We prompt GPT-5 to create 30 long story scripts spanning diverse styles, each containing a story overview, 8–12 shot-level text prompts, and scene-cut indicators.
In total, *ST-Bench* provides 300 detailed video prompts describing characters, scenes, dynamic events, shot types, and possibly camera movements.

To create your own story script, you can follow the system prompt below, which we use to generate structured, shot-level story scripts:

<details>
<summary><b>System prompt for story script (click to expand)</b></summary>


> You are an expert director of story videos. Your task is to design a story script about [..., e.g. a funny man].
> 
> Each prompt corresponds to a five-second video clip, so avoid overly complex text rendering, extreme motions, or audio-dependent effects. The overall story should remain simple, clear, and easy to follow.
>
> Your output must follow the JSON format shown in the example:
> 
> [ ... an example json story script here, e.g. `./story/black.json` ... ]
> 
> **Field Instructions**:
> 
> - **story_overview**: A concise summary of the whole story.
> - **scene_num**: Sequential index of the scene.
> - **cut**: Whether this prompt starts with a scene cut:
>   - `"True"`: a new cut.
>   - `"False"`: continue smoothly from the last frame of the previous prompt. Must ensure the two adjacent prompts can be naturally concatenated into a smooth continuous clip.
>   - The first prompt in the story must always have `"True"`.
> - **video_prompts**: A list of text-to-video prompts forming the story beats within the scene. Prompts should reflect natural, smooth, and logical story progression.
> 
> **What Each Video Prompt Should Describe (if relevant)**  
> 
> - **Characters**: appearance, attire, age, style.
> - **Actions & interactions**: motion, gestures, expressions, eye contact, simple physical actions.
> - **Scene & background**: indoor/outdoor location, props, layout, lighting, environment details.
> - **Atmosphere & mood**: emotional tone, colors, aesthetic feeling.
> - **Camera & editing**: shot type (e.g., close-up / medium / wide), simple camera movement, transitions.
>
> The prompts should be concise but sufficiently detailed (1–4 sentences).
>
> Return only a valid JSON story script.

</details>

To improve consistency, it would be better to provide
more explicit character descriptions in each shot prompt, helping the model match the intended memory.

## ❤️ Ackowledgement
Our implementation is based on [Wan2.2](https://github.com/Wan-Video/Wan2.2/tree/main). Thanks for the great open-source work!

## 📌 Citation
If any part of our paper or code is helpful to your research, please consider citing our work 📝 and give us a star ⭐. Thanks for your support!
```bibtex
@article{zhang2025storymem,
  title={{StoryMem}: Multi-shot Long Video Storytelling with Memory},
  author={Zhang, Kaiwen and Jiang, Liming and Wang, Angtian and Fang, Jacob Zhiyuan and Zhi, Tiancheng and Yan, Qing and Kang, Hao and Lu, Xin and Pan, Xingang},
  journal={arXiv preprint},
  volume={arXiv:2512.19539},
  year={2025}
}
```