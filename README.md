# Wan2.1 Text-to-Video Model

This repository contains the Wan2.1 text-to-video model, adapted for macOS with M1 Pro chip. This adaptation allows
macOS users to run the model while overcoming CUDA-specific limitations.

Note: This README contains part of original with some updates. AI Generated, please excuse oddities.

## Introduction

The Wan2.1 model is an open-source text-to-video generation model. It transforms textual descriptions into video
sequences, leveraging advanced machine learning techniques.

## Changes for macOS

This version includes modifications to make the model compatible with macOS, specifically for systems using the M1 Pro
chip. Key changes include:

- Adaptation of CUDA-specific code to work with MPS (Metal Performance Shaders) on macOS
- Environment variable settings for MPS fallback to CPU for unsupported operations
- Adjustments to command-line arguments for better compatibility with macOS

## Results

Here are some examples of videos generated using the macOS adaptation:

<video width="50%" controls autoplay muted loop><source src="/output/concatenated.mp4" type="video/mp4"></video>

<div style="display:flex; width: 100%; flex-wrap: wrap;">
<video width="49%" controls autoplay muted loop><source src="/output/jungle_walk_1.3b_3sec.mp4" type="video/mp4"></video>
<video width="49%" controls autoplay muted loop><source src="/output/jungle_walk_small_01.mp4" type="video/mp4"></video>
</div>

## Installation Instructions

Follow these steps to set up the environment on macOS:

1. **Install Homebrew**: If not already installed, use Homebrew to manage packages.
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.10+**:
   ```bash
   brew install python@3.13
   ```

3. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install einops
   ```

5. **Download models using huggingface-cli**:
   ```bash
   pip install "huggingface_hub[cli]"
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
   ```

## Usage

To generate a video, use the following command:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-1.3B --size "480*832" --frame_num 32 --sample_steps 15 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --device mps --prompt "Your prompt here" --save_file output_video.mp4
```

### Creating Longer Videos

For longer videos, we've included a script that generates multiple short segments and concatenates them:

```bash
python generate_loop.py --task t2v-1.3B --size "480*832" --frame_num 32 --sample_steps 15 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --device mps --prompt "Your prompt here" --num_videos 10 --concat
```

## Memory Optimization Tips

- **Use the 1.3B Model**: The 1.3B model works better than the 14B model on Mac hardware
- **Frame Count**: Keep frame counts moderate (32-48 frames) to avoid memory issues
- **Resolution**: 480x832 resolution works reliably
- **Memory Flags**: Use `--offload_model True` and `--t5_cpu` flags to reduce memory usage
- **CPU Fallback**: Enable MPS fallback to CPU for stability with `PYTORCH_ENABLE_MPS_FALLBACK=1`

## Acknowledgments

This project is based on the original Wan2.1 model. Special thanks to the original authors and contributors for their
work.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

* Feb 25, 2025: 👋 We've released the inference code and weights of Wan2.1.
* Feb 27, 2025: 👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/).
  Enjoy!

<div align="center">
  <video src="https://github.com/user-attachments/assets/4aca6063-60bf-4953-bfb7-e265053f49ef" width="70%" poster=""> </video>
</div>

## 📑 Todo List

- Wan2.1 Text-to-Video
    - [x] Multi-GPU Inference code of the 14B and 1.3B models
    - [x] Checkpoints of the 14B and 1.3B models
    - [x] Gradio demo
    - [x] ComfyUI integration
    - [ ] Diffusers integration
- Wan2.1 Image-to-Video
    - [x] Multi-GPU Inference code of the 14B model
    - [x] Checkpoints of the 14B model
    - [x] Gradio demo
    - [X] ComfyUI integration
    - [ ] Diffusers integration

| Models | Download Link | Notes |
|------------------------------------------------------------------------------|---------------------------------------------------------------------|
| T2V-14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)      | Supports both 480P and 720P |
| I2V-14B-720P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | Supports 720P |
| I2V-14B-480P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | Supports 480P |
| T2V-1.3B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)     | Supports 480P |

> 💡Note: The 1.3B model is capable of generating videos at 720P resolution. However, due to limited training at this
> resolution, the results are generally less stable compared to 480P. For optimal performance, we recommend using 480P
> resolution.


Download models using huggingface-cli:

```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

Download models using modelscope-cli:

```
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```

#### Run Text-to-Video Generation

This repository supports two Text-to-Video models (1.3B and 14B) and two resolutions (480P and 720P). The parameters and
configurations for these models are as follows:

<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>t2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-T2V-14B</td>
        </tr>
        <tr>
            <td>t2v-1.3B</td>
            <td style="color: green;">✔️</td>
            <td style="color: red;">❌</td>
            <td>Wan2.1-T2V-1.3B</td>
        </tr>
    </tbody>
</table>

##### (1) Without Prompt Extention

To facilitate implementation, we will start with a basic version of the inference process that skips
the [prompt extension](#2-using-prompt-extention) step.

- Single-GPU inference

```
python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True` and `--t5_cpu` options to reduce GPU
memory usage. For example, on an RTX 4090 GPU:

```
python generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡Note: If you are using the `T2V-1.3B` model, we recommend setting the parameter `--sample_guide_scale 6`. The
`--sample_shift parameter` can be adjusted within the range of 8 to 12 based on the performance.

- Multi-GPU inference using FSDP + xDiT USP

```
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

##### (2) Using Prompt Extention

Extending the prompts can effectively enrich the details in the generated videos, further enhancing the video quality.
Therefore, we recommend enabling prompt extension. We provide the following two methods for prompt extension:

## Usage

To generate a video, use the following command:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-1.3B --size "480*832" --frame_num 16 --sample_steps 25 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --device mps --prompt "Lion running under snow in Samarkand" --save_file output_video.mp4
```

DASH_API_KEY=your_key python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two
anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend
--prompt_extend_method 'dashscope' --prompt_extend_target_lang 'ch'

```

- Using a local model for extension.

  - By default, the Qwen model on HuggingFace is used for this extension. Users can choose Qwen models or other models based on the available GPU memory size.
  - For text-to-video tasks, you can use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.
  - For image-to-video tasks, you can use models like `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`.
  - Larger models generally provide better extension results but require more GPU memory.
  - You can modify the model used for extension with the parameter `--prompt_extend_model` , allowing you to specify either a local model path or a Hugging Face model. For example:

```

python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in
comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method '
local_qwen' --prompt_extend_target_lang 'ch'

```


#### Run Image-to-Video Generation

Similar to Text-to-Video, Image-to-Video is also divided into processes with and without the prompt extension step. The specific parameters and their corresponding settings are as follows:
<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>i2v-14B</td>
            <td style="color: green;">❌</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-I2V-14B-720P</td>
        </tr>
        <tr>
            <td>i2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: red;">❌</td>
            <td>Wan2.1-T2V-14B-480P</td>
        </tr>
    </tbody>
</table>


##### (1) Without Prompt Extention

- Single-GPU inference
```

python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline
gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring
crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed
posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and
the refreshing atmosphere of the seaside."

```

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Multi-GPU inference using FSDP + xDiT USP

```

pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image
examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing
sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred
beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white
clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot
highlights the feline's intricate details and the refreshing atmosphere of the seaside."

```

##### (2) Using Prompt Extention


The process of prompt extension can be referenced [here](#2-using-prompt-extention).

Run with local prompt extention using `Qwen/Qwen2.5-VL-7B-Instruct`:
```

python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG
--use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "Summer beach vacation style, a white cat
wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression.
Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted
with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A
close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

```

Run with remote prompt extention using `dashscope`:
```

DASH_API_KEY=your_key python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image
examples/i2v_input.JPG --use_prompt_extend --prompt_extend_method 'dashscope' --prompt "Summer beach vacation style, a
white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed
expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue
sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm
sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

```

##### (3) Runing local gradio

```

cd gradio

# if one only uses 480P model in gradio

DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_480p
./Wan2.1-I2V-14B-480P

# if one only uses 720P model in gradio

DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_720p
./Wan2.1-I2V-14B-720P

# if one uses both 480P and 720P models in gradio

DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_480p
./Wan2.1-I2V-14B-480P --ckpt_dir_720p ./Wan2.1-I2V-14B-720P

```


#### Run Text-to-Image Generation

Wan2.1 is a unified model for both image and video generation. Since it was trained on both types of data, it can also generate images. The command for generating images is similar to video generation, as follows:

##### (1) Without Prompt Extention

- Single-GPU inference
```

python generate.py --task t2i-14B --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B --prompt '一个朴素端庄的美人'

```

- Multi-GPU inference using FSDP + xDiT USP

```

torchrun --nproc_per_node=8 generate.py --dit_fsdp --t5_fsdp --ulysses_size 8 --base_seed 0 --frame_num 1 --task t2i-14B
--size 1024*1024 --prompt '一个朴素端庄的美人' --ckpt_dir ./Wan2.1-T2V-14B

```

##### (2) With Prompt Extention

- Single-GPU inference
```

python generate.py --task t2i-14B --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B --prompt '一个朴素端庄的美人'
--use_prompt_extend

```

- Multi-GPU inference using FSDP + xDiT USP
```

torchrun --nproc_per_node=8 generate.py --dit_fsdp --t5_fsdp --ulysses_size 8 --base_seed 0 --frame_num 1 --task t2i-14B
--size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B --prompt '一个朴素端庄的美人' --use_prompt_extend

```


## Manual Evaluation

##### (1) Text-to-Video Evaluation

Through manual evaluation, the results generated after prompt extension are superior to those from both closed-source and open-source models.

<div align="center">
    <img src="assets/t2v_res.jpg" alt="" style="width: 80%;" />
</div>


##### (2) Image-to-Video Evaluation

We also conducted extensive manual evaluations to evaluate the performance of the Image-to-Video model, and the results are presented in the table below. The results clearly indicate that **Wan2.1** outperforms both closed-source and open-source models.

<div align="center">
    <img src="assets/i2v_res.png" alt="" style="width: 80%;" />
</div>


## Computational Efficiency on Different GPUs

We test the computational efficiency of different **Wan2.1** models on different GPUs in the following table. The results are presented in the format: **Total time (s) / peak GPU memory (GB)**.


<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> The parameter settings for the tests presented in this table are as follows:
> (1) For the 1.3B model on 8 GPUs, set `--ring_size 8` and `--ulysses_size 1`;
> (2) For the 14B model on 1 GPU, use `--offload_model True`;
> (3) For the 1.3B model on a single 4090 GPU, set `--offload_model True --t5_cpu`;
> (4) For all testings, no prompt extension was applied, meaning `--use_prompt_extend` was not enabled.

> 💡Note: T2V-14B is slower than I2V-14B because the former samples 50 steps while the latter uses 40 steps.


## Community Contributions
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides more support for **Wan2.1**, including video-to-video, FP8 quantization, VRAM optimization, LoRA training, and more. Please refer to [their examples](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo).

-------

## Introduction of Wan2.1

**Wan2.1**  is designed on the mainstream diffusion transformer paradigm, achieving significant advancements in generative capabilities through a series of innovations. These include our novel spatio-temporal variational autoencoder (VAE), scalable training strategies, large-scale data construction, and automated evaluation metrics. Collectively, these contributions enhance the model's performance and versatility.


##### (1) 3D Variational Autoencoders
We propose a novel 3D causal VAE architecture, termed **Wan-VAE** specifically designed for video generation. By combining multiple strategies, we improve spatio-temporal compression, reduce memory usage, and ensure temporal causality. **Wan-VAE** demonstrates significant advantages in performance efficiency compared to other open-source VAEs. Furthermore, our **Wan-VAE** can encode and decode unlimited-length 1080P videos without losing historical temporal information, making it particularly well-suited for video generation tasks.


<div align="center">
    <img src="assets/video_vae_res.jpg" alt="" style="width: 80%;" />
</div>


##### (2) Video Diffusion DiT

**Wan2.1** is designed using the Flow Matching framework within the paradigm of mainstream Diffusion Transformers. Our model's architecture uses the T5 Encoder to encode multilingual text input, with cross-attention in each transformer block embedding the text into the model structure. Additionally, we employ an MLP with a Linear layer and a SiLU layer to process the input time embeddings and predict six modulation parameters individually. This MLP is shared across all transformer blocks, with each block learning a distinct set of biases. Our experimental findings reveal a significant performance improvement with this approach at the same parameter scale.

<div align="center">
    <img src="assets/video_dit_arch.jpg" alt="" style="width: 80%;" />
</div>


| Model  | Dimension | Input Dimension | Output Dimension | Feedforward Dimension | Frequency Dimension | Number of Heads | Number of Layers |
|--------|-----------|-----------------|------------------|-----------------------|---------------------|-----------------|------------------|
| 1.3B   | 1536      | 16              | 16               | 8960                  | 256                 | 12              | 30               |
| 14B   | 5120       | 16              | 16               | 13824                 | 256                 | 40              | 40               |



##### Data

We curated and deduplicated a candidate dataset comprising a vast amount of image and video data. During the data curation process, we designed a four-step data cleaning process, focusing on fundamental dimensions, visual quality and motion quality. Through the robust data processing pipeline, we can easily obtain high-quality, diverse, and large-scale training sets of images and videos.

![figure1](assets/data_for_diff_stage.jpg "figure1")


##### Comparisons to SOTA
We compared **Wan2.1** with leading open-source and closed-source models to evaluate the performace. Using our carefully designed set of 1,035 internal prompts, we tested across 14 major dimensions and 26 sub-dimensions. We then compute the total score by performing a weighted calculation on the scores of each dimension, utilizing weights derived from human preferences in the matching process. The detailed results are shown in the table below. These results demonstrate our model's superior performance compared to both open-source and closed-source models.

![figure1](assets/vben_vs_sota.png "figure1")


## Citation
If you find our work helpful, please cite us.

```

@article{wan2.1,
title = {Wan: Open and Advanced Large-Scale Video Generative Models},
author = {Wan Team},
journal = {},
year = {2025}
}

```

## License Agreement
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generate contents, granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. For a complete list of restrictions and details regarding your rights, please refer to the full text of the [license](LICENSE.txt).


## Acknowledgements


## Optimization Tips

- **Use CPU for Large Models**: If you encounter memory issues, use `--device cpu`.
- **Reduce Resolution and Frame Count**: Use smaller resolutions and fewer frames to reduce memory usage.
- **Monitor System Resources**: Keep an eye on memory usage and adjust parameters as needed.

## Acknowledgments

This project is based on the original Wan2.1 model. Special thanks to the original authors and contributors for their work.
