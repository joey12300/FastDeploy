# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os

from pipeline_stable_diffusion import StableDiffusionFastDeployPipeline
from scheduling_utils import PNDMScheduler
from paddlenlp.transformers import CLIPTokenizer

import fastdeploy as fd
from fastdeploy import ModelFormat
import numpy as np


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_model_dir",
        default="diffusion_model",
        help="The model directory of diffusion_model.")
    parser.add_argument(
        "--model_format",
        default="paddle",
        choices=['paddle', 'onnx'],
        help="The model format.")
    parser.add_argument(
        "--unet_model_prefix",
        default='unet',
        help="The file prefix of unet model.")
    parser.add_argument(
        "--vae_model_prefix",
        default='vae_decoder',
        help="The file prefix of vae model.")
    parser.add_argument(
        "--text_encoder_model_prefix",
        default='text_encoder',
        help="The file prefix of text_encoder model.")
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=100,
        help="The number of unet inference steps.")
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=1,
        help="The number of performance benchmark steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default='ort',
        choices=['ort', 'tensorrt', 'pp', 'pp-trt'],
        help="The inference runtime backend of unet model and text encoder model."
    )
    return parser.parse_args()


def create_ort_runtime(model_dir, model_prefix, model_format):
    option = fd.RuntimeOption()
    option.use_ort_backend()
    option.use_gpu()
    if model_format == "paddle":
        model_file = os.path.join(model_dir, f"{model_prefix}.pdmodel")
        params_file = os.path.join(model_dir, f"{model_prefix}.pdiparams")
        option.set_model_path(onnx_file, model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, f"{model_prefix}.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    return fd.Runtime(option)


def create_trt_runtime(model_dir,
                       model_prefix,
                       model_format,
                       workspace=(1 << 31),
                       dynamic_shape=None):
    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu()
    option.enable_trt_fp16()
    option.set_trt_max_workspace_size(workspace)
    if dynamic_shape is not None:
        for key, shape_dict in dynamic_shape.items():
            option.set_trt_input_shape(
                key,
                min_shape=shape_dict["min_shape"],
                opt_shape=shape_dict.get("opt_shape", None),
                max_shape=shape_dict.get("max_shape", None))
    if model_format == "paddle":
        model_file = os.path.join(model_dir, f"{model_prefix}.pdmodel")
        params_file = os.path.join(model_dir, f"{model_prefix}.pdiparams")
        option.set_model_path(onnx_file, model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, f"{model_prefix}.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    option.set_trt_cache_file(f"{model_prefix}.trt")
    return fd.Runtime(option)


if __name__ == "__main__":
    args = parse_arguments()
    # 1. Init scheduler
    scheduler = PNDMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        skip_prk_steps=True)

    # 2. Init tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 3. Init runtime
    text_encoder_runtime = create_ort_runtime(args.text_encoder_onnx_file)

    if args.backend == "onnx_runtime":
        vae_decoder_runtime = create_ort_runtime(args.vae_onnx_file)
        start = time.time()
        unet_runtime = create_ort_runtime(args.unet_onnx_file)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    else:
        vae_dynamic_shape = {
            "latent": {
                "min_shape": [1, 4, 64, 64],
                "max_shape": [2, 4, 64, 64],
                "opt_shape": [2, 4, 64, 64],
            }
        }
        vae_decoder_runtime = create_trt_runtime(
            args.vae_onnx_file,
            workspace=(1 << 30),
            dynamic_shape=vae_dynamic_shape)
        unet_dynamic_shape = {
            "latent_input": {
                "min_shape": [1, 4, 64, 64],
                "max_shape": [2, 4, 64, 64],
                "opt_shape": [2, 4, 64, 64],
            },
            "encoder_embedding": {
                "min_shape": [1, 77, 768],
                "max_shape": [2, 77, 768],
                "opt_shape": [2, 77, 768],
            },
        }
        start = time.time()
        unet_runtime = create_trt_runtime(
            args.unet_onnx_file, dynamic_shape=unet_dynamic_shape)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    pipe = StableDiffusionFastDeployPipeline(
        vae_decoder_runtime=vae_decoder_runtime,
        text_encoder_runtime=text_encoder_runtime,
        tokenizer=tokenizer,
        unet_runtime=unet_runtime,
        scheduler=scheduler)

    prompt = "a photo of an astronaut riding a horse on mars"
    # Warm up
    pipe(prompt, num_inference_steps=10)

    time_costs = []
    print(
        f"Run the stable diffusion pipeline {args.benchmark_steps} times to test the performance."
    )
    for step in range(args.benchmark_steps):
        start = time.time()
        image = pipe(prompt, num_inference_steps=args.inference_steps)[0]
        latency = time.time() - start
        time_costs += [latency]
        print(f"No {step:3d} time cost: {latency:2f} s")
    print(
        f"Mean latency: {np.mean(time_costs):2f}, p50 latency: {np.percentile(time_costs, 50):2f}, "
        f"p90 latency: {np.percentile(time_costs, 90):2f}, p95 latency: {np.percentile(time_costs, 95):2f}."
    )
    image.save("fd_astronaut_rides_horse.png")
    print(f"Image saved!")
