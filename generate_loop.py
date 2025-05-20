import argparse
import os
import sys
import logging
import torch
import random
from datetime import datetime
from PIL import Image
import subprocess
import glob

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# Import the original generate function
from generate import generate as original_generate

def generate_loop(args):
    """
    Generate multiple short videos in a loop, using the last frame of each video
    as a starting point for the next one.
    """
    # Create output directory if it doesn't exist
    output_dir = args.output_dir or "loop_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )
    
    # Initialize variables
    num_videos = args.num_videos
    base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    logging.info(f"Starting loop generation of {num_videos} videos with base seed {base_seed}")
    
    # Generate videos in a loop
    for i in range(num_videos):
        # Create a copy of args for this iteration
        current_args = argparse.Namespace(**vars(args))
        
        # Set the output file for this iteration
        current_args.save_file = os.path.join(output_dir, f"{i:04d}.mp4")
        
        # Set a unique seed for this iteration
        current_args.base_seed = base_seed + i
        
        # Generate the video
        logging.info(f"Generating video {i+1}/{num_videos} with seed {current_args.base_seed}")
        original_generate(current_args)
        
        logging.info(f"Video {i+1}/{num_videos} saved to {current_args.save_file}")
    
    # Concatenate all videos if requested
    if args.concat:
        concat_videos(output_dir, args.concat_output)
    
    logging.info("Loop generation completed")

def concat_videos(input_dir, output_file):
    """
    Concatenate all MP4 files in the input directory into a single video.
    """
    # Get all MP4 files in the directory
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    if not video_files:
        logging.error("No MP4 files found in the input directory")
        return
    
    # Create a file list for ffmpeg
    list_file = os.path.join(input_dir, "list.txt")
    with open(list_file, "w") as f:
        for video_file in video_files:
            f.write(f"file '{os.path.abspath(video_file)}'\n")
    
    # Concatenate videos using ffmpeg
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_file
    ]
    
    logging.info(f"Concatenating videos with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Clean up
    os.remove(list_file)
    
    logging.info(f"Concatenated video saved to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple short videos in a loop"
    )
    
    # Add all arguments from the original generate.py
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=32,
        help="How many frames to sample from a image or video."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=True,
        help="Whether to offload the model to CPU after each model forward."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT."
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT."
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5."
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU."
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from."
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend."
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use."
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use."
    )
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The base seed to use for generating the videos."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=15,
        help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use for computation (mps, cpu)."
    )
    
    # Add new arguments for the loop
    parser.add_argument(
        "--num_videos",
        type=int,
        default=10,
        help="Number of videos to generate in the loop."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="loop_output",
        help="Directory to save the generated videos."
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        default=False,
        help="Whether to concatenate all videos into a single file."
    )
    parser.add_argument(
        "--concat_output",
        type=str,
        default="concatenated.mp4",
        help="Output file for the concatenated video."
    )
    
    args = parser.parse_args()
    
    # Validate args
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupport size {args.size} for task {args.task}"
    
    return args

if __name__ == "__main__":
    args = parse_args()
    generate_loop(args) 