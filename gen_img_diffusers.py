# txt2img with Diffusers: supports SD checkpoints, EulerScheduler, clip-skip, 225 tokens, Hypernetwork etc...

# v2: CLIP guided Stable Diffusion, Image guided Stable Diffusion, highres. fix
# v3: Add dpmsolver/dpmsolver++, add VAE loading, add upscale, add 'bf16', fix the issue hypernetwork_mul is not working
# v4: SD2.0 support (new U-Net/text encoder/tokenizer), simplify by DiffUsers 0.9.0, no_preview in interactive mode


# Copyright 2022 kohya_ss @kohya_ss
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

# license of included scripts:

# FlashAttention: based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# Diffusers (model conversion, CLIP guided stable diffusion, schedulers etc.):
# ASL 2.0 https://github.com/huggingface/diffusers/blob/main/LICENSE


from typing import List, Optional, Union
import glob
import importlib
import inspect
import time
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import argparse
import math
import os
import random
import re
from typing import Any, Callable, List, Optional, Union

import diffusers
import numpy as np
import torch
from diffusers import (AutoencoderKL, DDPMScheduler,
                       EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
                       LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler, EulerDiscreteScheduler,
                       UNet2DConditionModel, StableDiffusionPipeline)
from einops import rearrange
from torch import einsum
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPTextConfig
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"     # ここからtokenizerだけ使う

DEFAULT_TOKEN_LENGTH = 75

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = 'scaled_linear'

LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

# CLIP guided SD関連
CLIP_MODEL_PATH = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FEATURE_EXTRACTOR_SIZE = (224, 224)
FEATURE_EXTRACTOR_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FEATURE_EXTRACTOR_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

# CLIP特徴量の取得時にcutoutを使うか：使う場合にはソースを書き換えてください
NUM_CUTOUTS = 4
USE_CUTOUTS = False

# region モデル変換

# StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 32  # unused
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024

# region checkpoint変換、読み込み、書き込み ###############################

# region StableDiffusion->Diffusersの変換コード
# convert_original_stable_diffusion_to_diffusers をコピーしている（ASL 2.0）


def shave_segments(path, n_shave_prefix_segments=1):
  """
  Removes segments. Positive values shave the first segments, negative shave the last segments.
  """
  if n_shave_prefix_segments >= 0:
    return ".".join(path.split(".")[n_shave_prefix_segments:])
  else:
    return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item.replace("in_layers.0", "norm1")
    new_item = new_item.replace("in_layers.2", "conv1")

    new_item = new_item.replace("out_layers.0", "norm2")
    new_item = new_item.replace("out_layers.3", "conv2")

    new_item = new_item.replace("emb_layers.1", "time_emb_proj")
    new_item = new_item.replace("skip_connection", "conv_shortcut")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("nin_shortcut", "conv_shortcut")
    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
    #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

    #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
    #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

    #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("norm.weight", "group_norm.weight")
    new_item = new_item.replace("norm.bias", "group_norm.bias")

    new_item = new_item.replace("q.weight", "query.weight")
    new_item = new_item.replace("q.bias", "query.bias")

    new_item = new_item.replace("k.weight", "key.weight")
    new_item = new_item.replace("k.bias", "key.bias")

    new_item = new_item.replace("v.weight", "value.weight")
    new_item = new_item.replace("v.bias", "value.bias")

    new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
    new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
  """
  This does the final conversion step: take locally converted weights and apply a global renaming
  to them. It splits attention layers, and takes into account additional replacements
  that may arise.

  Assigns the weights to the new checkpoint.
  """
  assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

  # Splits the attention layers into three variables.
  if attention_paths_to_split is not None:
    for path, path_map in attention_paths_to_split.items():
      old_tensor = old_checkpoint[path]
      channels = old_tensor.shape[0] // 3

      target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

      num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

      old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
      query, key, value = old_tensor.split(channels // num_heads, dim=1)

      checkpoint[path_map["query"]] = query.reshape(target_shape)
      checkpoint[path_map["key"]] = key.reshape(target_shape)
      checkpoint[path_map["value"]] = value.reshape(target_shape)

  for path in paths:
    new_path = path["new"]

    # These have already been assigned
    if attention_paths_to_split is not None and new_path in attention_paths_to_split:
      continue

    # Global renaming happens here
    new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
    new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
    new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

    if additional_replacements is not None:
      for replacement in additional_replacements:
        new_path = new_path.replace(replacement["old"], replacement["new"])

    # proj_attn.weight has to be converted from conv 1D to linear
    if "proj_attn.weight" in new_path:
      checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
    else:
      checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
  keys = list(checkpoint.keys())
  attn_keys = ["query.weight", "key.weight", "value.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in attn_keys:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0, 0]
    elif "proj_attn.weight" in key:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0]


def linear_transformer_to_conv(checkpoint):
  keys = list(checkpoint.keys())
  tf_keys = ["proj_in.weight", "proj_out.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in tf_keys:
      if checkpoint[key].ndim == 2:
        checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def convert_ldm_unet_checkpoint(v2, checkpoint, config):
  """
  Takes a state dict and a config, and returns a converted checkpoint.
  """

  # extract state_dict for UNet
  unet_state_dict = {}
  unet_key = "model.diffusion_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(unet_key):
      unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

  new_checkpoint = {}

  new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
  new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
  new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
  new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

  new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
  new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

  new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
  new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
  new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
  new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

  # Retrieves the keys for the input blocks only
  num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
  input_blocks = {
      layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
      for layer_id in range(num_input_blocks)
  }

  # Retrieves the keys for the middle blocks only
  num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
  middle_blocks = {
      layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
      for layer_id in range(num_middle_blocks)
  }

  # Retrieves the keys for the output blocks only
  num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
  output_blocks = {
      layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
      for layer_id in range(num_output_blocks)
  }

  for i in range(1, num_input_blocks):
    block_id = (i - 1) // (config["layers_per_block"] + 1)
    layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

    resnets = [
        key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
    ]
    attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

    if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.weight"
      )
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.bias"
      )

    paths = renew_resnet_paths(resnets)
    meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
    assign_to_checkpoint(
        paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    if len(attentions):
      paths = renew_attention_paths(attentions)
      meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

  resnet_0 = middle_blocks[0]
  attentions = middle_blocks[1]
  resnet_1 = middle_blocks[2]

  resnet_0_paths = renew_resnet_paths(resnet_0)
  assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

  resnet_1_paths = renew_resnet_paths(resnet_1)
  assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

  attentions_paths = renew_attention_paths(attentions)
  meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(
      attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
  )

  for i in range(num_output_blocks):
    block_id = i // (config["layers_per_block"] + 1)
    layer_in_block_id = i % (config["layers_per_block"] + 1)
    output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
    output_block_list = {}

    for layer in output_block_layers:
      layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
      if layer_id in output_block_list:
        output_block_list[layer_id].append(layer_name)
      else:
        output_block_list[layer_id] = [layer_name]

    if len(output_block_list) > 1:
      resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
      attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

      resnet_0_paths = renew_resnet_paths(resnets)
      paths = renew_resnet_paths(resnets)

      meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

      if ["conv.weight", "conv.bias"] in output_block_list.values():
        index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.weight"
        ]
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.bias"
        ]

        # Clear attentions as they have been attributed above.
        if len(attentions) == 2:
          attentions = []

      if len(attentions):
        paths = renew_attention_paths(attentions)
        meta_path = {
            "old": f"output_blocks.{i}.1",
            "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )
    else:
      resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
      for path in resnet_0_paths:
        old_path = ".".join(["output_blocks", str(i), path["old"]])
        new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

        new_checkpoint[new_path] = unet_state_dict[old_path]

  # SDのv2では1*1のconv2dがlinearに変わっているので、linear->convに変換する
  if v2:
    linear_transformer_to_conv(new_checkpoint)

  return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
  # extract state dict for VAE
  vae_state_dict = {}
  vae_key = "first_stage_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(vae_key):
      vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

  new_checkpoint = {}

  new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
  new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
  new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
  new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
  new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
  new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

  new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
  new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
  new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
  new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
  new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
  new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

  new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
  new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
  new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
  new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

  # Retrieves the keys for the encoder down blocks only
  num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
  down_blocks = {
      layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
  }

  # Retrieves the keys for the decoder up blocks only
  num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
  up_blocks = {
      layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
  }

  for i in range(num_down_blocks):
    resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

    if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.weight"
      )
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.bias"
      )

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)

  for i in range(num_up_blocks):
    block_id = num_up_blocks - 1 - i
    resnets = [
        key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
    ]

    if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.weight"
      ]
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.bias"
      ]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)
  return new_checkpoint


def create_unet_diffusers_config(v2):
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # unet_params = original_config.model.params.unet_config.params

  block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

  down_block_types = []
  resolution = 1
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
    down_block_types.append(block_type)
    if i != len(block_out_channels) - 1:
      resolution *= 2

  up_block_types = []
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
    up_block_types.append(block_type)
    resolution //= 2

  config = dict(
      sample_size=UNET_PARAMS_IMAGE_SIZE,
      in_channels=UNET_PARAMS_IN_CHANNELS,
      out_channels=UNET_PARAMS_OUT_CHANNELS,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
      cross_attention_dim=UNET_PARAMS_CONTEXT_DIM if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
      attention_head_dim=UNET_PARAMS_NUM_HEADS if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
  )

  return config


def create_vae_diffusers_config():
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # vae_params = original_config.model.params.first_stage_config.params.ddconfig
  # _ = original_config.model.params.first_stage_config.params.embed_dim
  block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
  down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
  up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

  config = dict(
      sample_size=VAE_PARAMS_RESOLUTION,
      in_channels=VAE_PARAMS_IN_CHANNELS,
      out_channels=VAE_PARAMS_OUT_CH,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      latent_channels=VAE_PARAMS_Z_CHANNELS,
      layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
  )
  return config


def convert_ldm_clip_checkpoint_v1(checkpoint):
  keys = list(checkpoint.keys())
  text_model_dict = {}
  for key in keys:
    if key.startswith("cond_stage_model.transformer"):
      text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]
  return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
  # 嫌になるくらい違うぞ！
  def convert_key(key):
    if not key.startswith("cond_stage_model"):
      return None

    # common conversion
    key = key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
    key = key.replace("cond_stage_model.model.", "text_model.")

    if "resblocks" in key:
      # resblocks conversion
      key = key.replace(".resblocks.", ".layers.")
      if ".ln_" in key:
        key = key.replace(".ln_", ".layer_norm")
      elif ".mlp." in key:
        key = key.replace(".c_fc.", ".fc1.")
        key = key.replace(".c_proj.", ".fc2.")
      elif '.attn.out_proj' in key:
        key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
      elif '.attn.in_proj' in key:
        key = None                  # 特殊なので後で処理する
      else:
        raise ValueError(f"unexpected key in SD: {key}")
    elif '.positional_embedding' in key:
      key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
    elif '.text_projection' in key:
      key = None    # 使われない???
    elif '.logit_scale' in key:
      key = None    # 使われない???
    elif '.token_embedding' in key:
      key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
    elif '.ln_final' in key:
      key = key.replace(".ln_final", ".final_layer_norm")
    return key

  keys = list(checkpoint.keys())
  new_sd = {}
  for key in keys:
    # remove resblocks 23
    if '.resblocks.23.' in key:
      continue
    new_key = convert_key(key)
    if new_key is None:
      continue
    new_sd[new_key] = checkpoint[key]

  # attnの変換
  for key in keys:
    if '.resblocks.23.' in key:
      continue
    if '.resblocks' in key and '.attn.in_proj_' in key:
      # 三つに分割
      values = torch.chunk(checkpoint[key], 3)

      key_suffix = ".weight" if "weight" in key else ".bias"
      key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
      key_pfx = key_pfx.replace("_weight", "")
      key_pfx = key_pfx.replace("_bias", "")
      key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
      new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
      new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
      new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

  # position_idsの追加
  new_sd["text_model.embeddings.position_ids"] = torch.Tensor([list(range(max_length))]).to(torch.int64)
  return new_sd

# endregion


# region Diffusers->StableDiffusion の変換コード
# convert_diffusers_to_original_stable_diffusion をコピーしている（ASL 2.0）

def conv_transformer_to_linear(checkpoint):
  keys = list(checkpoint.keys())
  tf_keys = ["proj_in.weight", "proj_out.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in tf_keys:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0, 0]


def convert_unet_state_dict_to_sd(v2, unet_state_dict):
  unet_conversion_map = [
      # (stable-diffusion, HF Diffusers)
      ("time_embed.0.weight", "time_embedding.linear_1.weight"),
      ("time_embed.0.bias", "time_embedding.linear_1.bias"),
      ("time_embed.2.weight", "time_embedding.linear_2.weight"),
      ("time_embed.2.bias", "time_embedding.linear_2.bias"),
      ("input_blocks.0.0.weight", "conv_in.weight"),
      ("input_blocks.0.0.bias", "conv_in.bias"),
      ("out.0.weight", "conv_norm_out.weight"),
      ("out.0.bias", "conv_norm_out.bias"),
      ("out.2.weight", "conv_out.weight"),
      ("out.2.bias", "conv_out.bias"),
  ]

  unet_conversion_map_resnet = [
      # (stable-diffusion, HF Diffusers)
      ("in_layers.0", "norm1"),
      ("in_layers.2", "conv1"),
      ("out_layers.0", "norm2"),
      ("out_layers.3", "conv2"),
      ("emb_layers.1", "time_emb_proj"),
      ("skip_connection", "conv_shortcut"),
  ]

  unet_conversion_map_layer = []
  for i in range(4):
      # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
      hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
      sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
      unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

      if i < 3:
        # no attention layers in down_blocks.3
        hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
        sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
        unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
      # loop over resnets/attentions for upblocks
      hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
      sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
      unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

      if i > 0:
        # no attention layers in up_blocks.0
        hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
        sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
        unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
      # no downsample in down_blocks.3
      hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
      sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
      unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

      # no upsample in up_blocks.3
      hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
      sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
      unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

  hf_mid_atn_prefix = "mid_block.attentions.0."
  sd_mid_atn_prefix = "middle_block.1."
  unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

  for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

  # buyer beware: this is a *brittle* function,
  # and correct output requires that all of these pieces interact in
  # the exact order in which I have arranged them.
  mapping = {k: k for k in unet_state_dict.keys()}
  for sd_name, hf_name in unet_conversion_map:
    mapping[hf_name] = sd_name
  for k, v in mapping.items():
    if "resnets" in k:
      for sd_part, hf_part in unet_conversion_map_resnet:
        v = v.replace(hf_part, sd_part)
      mapping[k] = v
  for k, v in mapping.items():
    for sd_part, hf_part in unet_conversion_map_layer:
      v = v.replace(hf_part, sd_part)
    mapping[k] = v
  new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

  if v2:
    conv_transformer_to_linear(new_state_dict)

  return new_state_dict

# endregion


def load_checkpoint_with_text_encoder_conversion(ckpt_path):
  # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
  TEXT_ENCODER_KEY_REPLACEMENTS = [
      ('cond_stage_model.transformer.embeddings.', 'cond_stage_model.transformer.text_model.embeddings.'),
      ('cond_stage_model.transformer.encoder.', 'cond_stage_model.transformer.text_model.encoder.'),
      ('cond_stage_model.transformer.final_layer_norm.', 'cond_stage_model.transformer.text_model.final_layer_norm.')
  ]

  checkpoint = torch.load(ckpt_path, map_location="cpu")
  state_dict = checkpoint["state_dict"]

  key_reps = []
  for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
    for key in state_dict.keys():
      if key.startswith(rep_from):
        new_key = rep_to + key[len(rep_from):]
        key_reps.append((key, new_key))

  for key, new_key in key_reps:
    state_dict[new_key] = state_dict[key]
    del state_dict[key]

  return checkpoint


def load_models_from_stable_diffusion_checkpoint(v2, ckpt_path, dtype=None):
  checkpoint = load_checkpoint_with_text_encoder_conversion(ckpt_path)
  state_dict = checkpoint["state_dict"]
  if dtype is not None:
    for k, v in state_dict.items():
      if type(v) is torch.Tensor:
        state_dict[k] = v.to(dtype)

  # Convert the UNet2DConditionModel model.
  unet_config = create_unet_diffusers_config(v2)
  converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

  unet = UNet2DConditionModel(**unet_config)
  info = unet.load_state_dict(converted_unet_checkpoint)
  print("loading u-net:", info)

  # Convert the VAE model.
  vae_config = create_vae_diffusers_config()
  converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

  vae = AutoencoderKL(**vae_config)
  info = vae.load_state_dict(converted_vae_checkpoint)
  print("loadint vae:", info)

  # convert text_model
  if v2:
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
    cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=23,
        num_attention_heads=16,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=512,
        torch_dtype="float32",
        transformers_version="4.25.0.dev0",
    )
    text_model = CLIPTextModel._from_config(cfg)
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
  else:
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
  print("loading text encoder:", info)

  return text_model, vae, unet


def convert_text_encoder_state_dict_to_sd_v2(checkpoint):
  def convert_key(key):
    # position_idsの除去
    if ".position_ids" in key:
      return None

    # common
    key = key.replace("text_model.encoder.", "transformer.")
    key = key.replace("text_model.", "")
    if "layers" in key:
      # resblocks conversion
      key = key.replace(".layers.", ".resblocks.")
      if ".layer_norm" in key:
        key = key.replace(".layer_norm", ".ln_")
      elif ".mlp." in key:
        key = key.replace(".fc1.", ".c_fc.")
        key = key.replace(".fc2.", ".c_proj.")
      elif '.self_attn.out_proj' in key:
        key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
      elif '.self_attn.' in key:
        key = None                  # 特殊なので後で処理する
      else:
        raise ValueError(f"unexpected key in DiffUsers model: {key}")
    elif '.position_embedding' in key:
      key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
    elif '.token_embedding' in key:
      key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
    elif 'final_layer_norm' in key:
      key = key.replace("final_layer_norm", "ln_final")
    return key

  keys = list(checkpoint.keys())
  new_sd = {}
  for key in keys:
    new_key = convert_key(key)
    if new_key is None:
      continue
    new_sd[new_key] = checkpoint[key]

  # attnの変換
  for key in keys:
    if 'layers' in key and 'q_proj' in key:
      # 三つを結合
      key_q = key
      key_k = key.replace("q_proj", "k_proj")
      key_v = key.replace("q_proj", "v_proj")

      value_q = checkpoint[key_q]
      value_k = checkpoint[key_k]
      value_v = checkpoint[key_v]
      value = torch.cat([value_q, value_k, value_v])

      new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
      new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
      new_sd[new_key] = value

  return new_sd


def save_stable_diffusion_checkpoint(v2, output_file, text_encoder, unet, ckpt_path, epochs, steps, save_dtype=None):
  # VAEがメモリ上にないので、もう一度VAEを含めて読み込む
  checkpoint = load_checkpoint_with_text_encoder_conversion(ckpt_path)
  state_dict = checkpoint["state_dict"]

  def assign_new_sd(prefix, sd):
    for k, v in sd.items():
      key = prefix + k
      assert key in state_dict, f"Illegal key in save SD: {key}"
      if save_dtype is not None:
        v = v.detach().clone().to("cpu").to(save_dtype)
      state_dict[key] = v

  # Convert the UNet model
  unet_state_dict = convert_unet_state_dict_to_sd(v2, unet.state_dict())
  assign_new_sd("model.diffusion_model.", unet_state_dict)

  # Convert the text encoder model
  if v2:
    text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder.state_dict())
    assign_new_sd("cond_stage_model.model.", text_enc_dict)
  else:
    text_enc_dict = text_encoder.state_dict()
    assign_new_sd("cond_stage_model.transformer.", text_enc_dict)

  # Put together new checkpoint
  new_ckpt = {'state_dict': state_dict}

  if 'epoch' in checkpoint:
    epochs += checkpoint['epoch']
  if 'global_step' in checkpoint:
    steps += checkpoint['global_step']

  new_ckpt['epoch'] = epochs
  new_ckpt['global_step'] = steps

  torch.save(new_ckpt, output_file)


def save_diffusers_checkpoint(v2, output_dir, text_encoder, unet, pretrained_model_name_or_path, save_dtype):
  pipeline = StableDiffusionPipeline(
      unet=unet,
      text_encoder=text_encoder,
      vae=AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae"),
      scheduler=DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler"),
      tokenizer=CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer"),
      safety_checker=None,
      feature_extractor=None,
      requires_safety_checker=None,
  )
  pipeline.save_pretrained(output_dir)

# endregion


# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
  return val is not None


def default(val, d):
  return val if exists(val) else d

# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.Function):
  @ staticmethod
  @ torch.no_grad()
  def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
    """ Algorithm 2 in the paper """

    device = q.device
    dtype = q.dtype
    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    o = torch.zeros_like(q)
    all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
    all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

    scale = (q.shape[-1] ** -0.5)

    if not exists(mask):
      mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
    else:
      mask = rearrange(mask, 'b n -> b 1 1 n')
      mask = mask.split(q_bucket_size, dim=-1)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        mask,
        all_row_sums.split(q_bucket_size, dim=-2),
        all_row_maxes.split(q_bucket_size, dim=-2),
    )

    for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if exists(row_mask):
          attn_weights.masked_fill_(~row_mask, max_neg_value)

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
        attn_weights -= block_row_maxes
        exp_weights = torch.exp(attn_weights)

        if exists(row_mask):
          exp_weights.masked_fill_(~row_mask, 0.)

        block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

        new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

        exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

        exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
        exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

        new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

        oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

        row_maxes.copy_(new_row_maxes)
        row_sums.copy_(new_row_sums)

    ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
    ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

    return o

  @ staticmethod
  @ torch.no_grad()
  def backward(ctx, do):
    """ Algorithm 4 in the paper """

    causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
    q, k, v, o, l, m = ctx.saved_tensors

    device = q.device

    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        do.split(q_bucket_size, dim=-2),
        mask,
        l.split(q_bucket_size, dim=-2),
        m.split(q_bucket_size, dim=-2),
        dq.split(q_bucket_size, dim=-2)
    )

    for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
          dk.split(k_bucket_size, dim=-2),
          dv.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        exp_attn_weights = torch.exp(attn_weights - mc)

        if exists(row_mask):
          exp_attn_weights.masked_fill_(~row_mask, 0.)

        p = exp_attn_weights / lc

        dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
        dp = einsum('... i d, ... j d -> ... i j', doc, vc)

        D = (doc * oc).sum(dim=-1, keepdims=True)
        ds = p * scale * (dp - D)

        dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
        dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

        dqc.add_(dq_chunk)
        dkc.add_(dk_chunk)
        dvc.add_(dv_chunk)

    return dq, dk, dv, None, None, None, None


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
  if mem_eff_attn:
    replace_unet_cross_attn_to_memory_efficient()
  elif xformers:
    replace_unet_cross_attn_to_xformers()


def replace_unet_cross_attn_to_memory_efficient():
  print("Replace CrossAttention.forward to use Hypernetwork and FlashAttention")
  flash_func = FlashAttentionFunction

  def forward_flash_attn(self, x, context=None, mask=None):
    q_bucket_size = 512
    k_bucket_size = 1024

    h = self.heads
    q = self.to_q(x)

    context = context if context is not None else x
    context = context.to(x.dtype)

    if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
      context_k, context_v = self.hypernetwork.forward(x, context)
      context_k = context_k.to(x.dtype)
      context_v = context_v.to(x.dtype)
    else:
      context_k = context
      context_v = context

    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

    out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

    out = rearrange(out, 'b h n d -> b n (h d)')

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
  print("Replace CrossAttention.forward to use Hypernetwork and xformers")
  try:
    import xformers.ops
  except ImportError:
    raise ImportError("No xformers / xformersがインストールされていないようです")

  def forward_xformers(self, x, context=None, mask=None):
    h = self.heads
    q_in = self.to_q(x)

    context = default(context, x)
    context = context.to(x.dtype)

    if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
      context_k, context_v = self.hypernetwork.forward(x, context)
      context_k = context_k.to(x.dtype)
      context_v = context_v.to(x.dtype)
    else:
      context_k = context
      context_v = context

    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)        # 最適なのを選んでくれる

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_xformers
# endregion

# region 画像生成の本体：lpw_stable_diffusion.py （ASL）からコピーして修正
# https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py
# Pipelineだけ独立して使えないのと機能追加するのとでコピーして修正


class PipelineLike():
  r"""
  Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
  weighting in prompt.
  This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
  library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
  Args:
      vae ([`AutoencoderKL`]):
          Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
      text_encoder ([`CLIPTextModel`]):
          Frozen text-encoder. Stable Diffusion uses the text portion of
          [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
          the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
      tokenizer (`CLIPTokenizer`):
          Tokenizer of class
          [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
      unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
      scheduler ([`SchedulerMixin`]):
          A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
          [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
      safety_checker ([`StableDiffusionSafetyChecker`]):
          Classification module that estimates whether generated images could be considered offensive or harmful.
          Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
      feature_extractor ([`CLIPFeatureExtractor`]):
          Model that extracts features from generated images to be used as inputs for the `safety_checker`.
  """

  def __init__(
      self,
      device,
      vae: AutoencoderKL,
      text_encoder: CLIPTextModel,
      tokenizer: CLIPTokenizer,
      unet: UNet2DConditionModel,
      scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
      clip_skip: int,
      clip_model: CLIPModel,
      clip_guidance_scale: float,
      clip_image_guidance_scale: float,
      # safety_checker: StableDiffusionSafetyChecker,
      # feature_extractor: CLIPFeatureExtractor,
  ):
    super().__init__()
    self.device = device
    self.clip_skip = clip_skip

    if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
      deprecation_message = (
          f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
          f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
          "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
          " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
          " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
          " file"
      )
      deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
      new_config = dict(scheduler.config)
      new_config["steps_offset"] = 1
      scheduler._internal_dict = FrozenDict(new_config)

    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
      deprecation_message = (
          f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
          " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
          " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
          " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
          " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
      )
      deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
      new_config = dict(scheduler.config)
      new_config["clip_sample"] = False
      scheduler._internal_dict = FrozenDict(new_config)

    self.vae = vae
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self.unet = unet
    self.scheduler = scheduler
    self.safety_checker = None

    # CLIP guidance
    self.clip_guidance_scale = clip_guidance_scale
    self.clip_image_guidance_scale = clip_image_guidance_scale
    self.clip_model = clip_model
    self.normalize = transforms.Normalize(mean=FEATURE_EXTRACTOR_IMAGE_MEAN, std=FEATURE_EXTRACTOR_IMAGE_STD)
    self.make_cutouts = MakeCutouts(FEATURE_EXTRACTOR_SIZE)

# region xformersとか使う部分：独自に書き換えるので関係なし
  def enable_xformers_memory_efficient_attention(self):
    r"""
    Enable memory efficient attention as implemented in xformers.
    When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
    time. Speed up at training time is not guaranteed.
    Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
    is used.
    """
    self.unet.set_use_memory_efficient_attention_xformers(True)

  def disable_xformers_memory_efficient_attention(self):
    r"""
    Disable memory efficient attention as implemented in xformers.
    """
    self.unet.set_use_memory_efficient_attention_xformers(False)

  def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
    r"""
    Enable sliced attention computation.
    When this option is enabled, the attention module will split the input tensor in slices, to compute attention
    in several steps. This is useful to save some memory in exchange for a small speed decrease.
    Args:
        slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
            When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
            a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
            `attention_head_dim` must be a multiple of `slice_size`.
    """
    if slice_size == "auto":
      # half the attention head size is usually a good trade-off between
      # speed and memory
      slice_size = self.unet.config.attention_head_dim // 2
    self.unet.set_attention_slice(slice_size)

  def disable_attention_slicing(self):
    r"""
    Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
    back to computing attention in one step.
    """
    # set slice_size = `None` to disable `attention slicing`
    self.enable_attention_slicing(None)

  def enable_sequential_cpu_offload(self):
    r"""
    Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
    text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
    `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
    """
    # accelerateが必要になるのでとりあえず省略
    raise NotImplementedError("cpu_offload is omitted.")
    # if is_accelerate_available():
    #   from accelerate import cpu_offload
    # else:
    #   raise ImportError("Please install accelerate via `pip install accelerate`")

    # device = self.device

    # for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.safety_checker]:
    #   if cpu_offloaded_model is not None:
    #     cpu_offload(cpu_offloaded_model, device)
# endregion

  @torch.no_grad()
  def __call__(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
      mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
      height: int = 512,
      width: int = 512,
      num_inference_steps: int = 50,
      guidance_scale: float = 7.5,
      strength: float = 0.8,
      # num_images_per_prompt: Optional[int] = 1,
      eta: float = 0.0,
      generator: Optional[torch.Generator] = None,
      latents: Optional[torch.FloatTensor] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      # return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      is_cancelled_callback: Optional[Callable[[], bool]] = None,
      callback_steps: Optional[int] = 1,
      img2img_noise=None,
      clip_prompts=None,
      clip_guide_images=None,
      **kwargs,
  ):
    r"""
    Function invoked when calling the pipeline for generation.
    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        init_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, that will be used as the starting point for the
            process.
        mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
            replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
            PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
            contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
            `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
            number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
            noise will be maximum and the denoising process will run for the full number of iterations specified in
            `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        is_cancelled_callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. If the function returns
            `True`, the inference will be cancelled.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        `None` if cancelled by `is_cancelled_callback`,
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    num_images_per_prompt = 1              # fixed

    if isinstance(prompt, str):
      batch_size = 1
      prompt = [prompt]
    elif isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if strength < 0 or strength > 1:
      raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

    if height % 8 != 0 or width % 8 != 0:
      raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
        callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
      raise ValueError(
          f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
          f" {type(callback_steps)}."
      )

    # get prompt text embeddings

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if negative_prompt is None:
      negative_prompt = [""] * batch_size
    elif isinstance(negative_prompt, str):
      negative_prompt = [negative_prompt] * batch_size
    if batch_size != len(negative_prompt):
      raise ValueError(
          f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
          f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
          " the batch size of `prompt`."
      )

    text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
        pipe=self,
        prompt=prompt,
        uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
        max_embeddings_multiples=max_embeddings_multiples,
        clip_skip=self.clip_skip,
        **kwargs,
    )

    if do_classifier_free_guidance:
      text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # CLIP guidanceで使用するembeddingsを取得する
    if self.clip_guidance_scale > 0:
      clip_text_input = prompt_tokens
      if clip_text_input.shape[1] > self.tokenizer.model_max_length:
          # TODO 75文字を超えたら警告を出す？
        print("trim text input", clip_text_input.shape)
        clip_text_input = torch.cat([clip_text_input[:, :self.tokenizer.model_max_length-1],
                                    clip_text_input[:, -1].unsqueeze(1)], dim=1)
        print("trimmed", clip_text_input.shape)

      for i, clip_prompt in enumerate(clip_prompts):
        if clip_prompt is not None:         # clip_promptがあれば上書きする
          clip_text_input[i] = self.tokenizer(clip_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                              truncation=True, return_tensors="pt",).input_ids.to(self.device)

      text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
      text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)      # prompt複数件でもOK
    if self.clip_image_guidance_scale > 0:
      if isinstance(clip_guide_images, PIL.Image.Image):
        clip_guide_images = [clip_guide_images]
      clip_guide_images = [preprocess_guide_image(im) for im in clip_guide_images]
      clip_guide_images = torch.cat(clip_guide_images, dim=0)
      clip_guide_images = self.normalize(clip_guide_images).to(self.device).to(text_embeddings.dtype)

      image_embeddings_clip = self.clip_model.get_image_features(clip_guide_images)
      image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

      if len(image_embeddings_clip) == 1:
        image_embeddings_clip = image_embeddings_clip.repeat((batch_size, 1, 1, 1))

    # set timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    latents_dtype = text_embeddings.dtype
    init_latents_orig = None
    mask = None
    noise = None

    if init_image is None:
      # get the initial random noise unless the user supplied it

      # Unlike in other pipelines, latents need to be generated in the target device
      # for 1-to-1 results reproducibility with the CompVis implementation.
      # However this currently doesn't work in `mps`.
      latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8,)

      if latents is None:
        if self.device.type == "mps":
          # randn does not exist on mps
          latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype,).to(self.device)
        else:
          latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype,)
      else:
        if latents.shape != latents_shape:
          raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        latents = latents.to(self.device)

      timesteps = self.scheduler.timesteps.to(self.device)

      # scale the initial noise by the standard deviation required by the scheduler
      latents = latents * self.scheduler.init_noise_sigma
    else:
      # image to tensor
      if isinstance(init_image, PIL.Image.Image):
        init_image = [init_image]
      if isinstance(init_image[0], PIL.Image.Image):
        init_image = [preprocess_image(im) for im in init_image]
        init_image = torch.cat(init_image)

      # mask image to tensor
      if mask_image is not None:
        if isinstance(mask_image, PIL.Image.Image):
          mask_image = [mask_image]
        if isinstance(mask_image[0], PIL.Image.Image):
          mask_image = torch.cat([preprocess_mask(im) for im in mask_image])            # H*W, 0 for repaint

      # encode the init image into latents and scale the latents
      init_image = init_image.to(device=self.device, dtype=latents_dtype)
      init_latent_dist = self.vae.encode(init_image).latent_dist
      init_latents = init_latent_dist.sample(generator=generator)
      init_latents = 0.18215 * init_latents
      if len(init_latents) == 1:
        init_latents = init_latents.repeat((batch_size, 1, 1, 1))
      init_latents_orig = init_latents

      # preprocess mask
      if mask_image is not None:
        mask = mask_image.to(device=self.device, dtype=latents_dtype)
        if len(mask) == 1:
          mask = mask.repeats((batch_size, 1, 1, 1))

        # check sizes
        if not mask.shape == init_latents.shape:
          raise ValueError("The mask and init_image should be the same size!")

      # get the original timestep using init_timestep
      offset = self.scheduler.config.get("steps_offset", 0)
      init_timestep = int(num_inference_steps * strength) + offset
      init_timestep = min(init_timestep, num_inference_steps)

      timesteps = self.scheduler.timesteps[-init_timestep]
      timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=self.device)

      # add noise to latents using the timesteps
      latents = self.scheduler.add_noise(init_latents, img2img_noise, timesteps)

      t_start = max(num_inference_steps - init_timestep + offset, 0)
      timesteps = self.scheduler.timesteps[t_start:].to(self.device)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
      extra_step_kwargs["eta"] = eta

    for i, t in enumerate(tqdm(timesteps)):
      # expand the latents if we are doing classifier free guidance
      latent_model_input = latents.repeat((2, 1, 1, 1)) if do_classifier_free_guidance else latents
      latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

      # predict the noise residual
      noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

      # perform guidance
      if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      # perform clip guidance
      if self.clip_guidance_scale > 0 or self.clip_image_guidance_scale > 0:
        text_embeddings_for_guidance = (text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings)

        if self.clip_guidance_scale > 0:
          noise_pred, latents = self.cond_fn(latents, t, i, text_embeddings_for_guidance, noise_pred,
                                             text_embeddings_clip, self.clip_guidance_scale, NUM_CUTOUTS, USE_CUTOUTS,)
        if self.clip_image_guidance_scale > 0:
          noise_pred, latents = self.cond_fn(latents, t, i, text_embeddings_for_guidance, noise_pred,
                                             image_embeddings_clip, self.clip_image_guidance_scale, NUM_CUTOUTS, USE_CUTOUTS,)

      # compute the previous noisy sample x_t -> x_t-1
      latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

      if mask is not None:
        # masking
        init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
        latents = (init_latents_proper * mask) + (latents * (1 - mask))

      # call the callback, if provided
      if i % callback_steps == 0:
        if callback is not None:
          callback(i, t, latents)
        if is_cancelled_callback is not None and is_cancelled_callback():
          return None

    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    if self.safety_checker is not None:
      safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
          self.device
      )
      image, has_nsfw_concept = self.safety_checker(
          images=image,
          clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype),
      )
    else:
      has_nsfw_concept = None

    if output_type == "pil":
      # image = self.numpy_to_pil(image)
      image = (image * 255).round().astype("uint8")
      image = [Image.fromarray(im) for im in image]

    # if not return_dict:
    return (image, has_nsfw_concept)

    # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

  def text2img(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: int = 512,
      width: int = 512,
      num_inference_steps: int = 50,
      guidance_scale: float = 7.5,
      num_images_per_prompt: Optional[int] = 1,
      eta: float = 0.0,
      generator: Optional[torch.Generator] = None,
      latents: Optional[torch.FloatTensor] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      callback_steps: Optional[int] = 1,
      **kwargs,
  ):
    r"""
    Function for text-to-image generation.
    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    return self.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        latents=latents,
        max_embeddings_multiples=max_embeddings_multiples,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        **kwargs,
    )

  def img2img(
      self,
      init_image: Union[torch.FloatTensor, PIL.Image.Image],
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      strength: float = 0.8,
      num_inference_steps: Optional[int] = 50,
      guidance_scale: Optional[float] = 7.5,
      num_images_per_prompt: Optional[int] = 1,
      eta: Optional[float] = 0.0,
      generator: Optional[torch.Generator] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      callback_steps: Optional[int] = 1,
      **kwargs,
  ):
    r"""
    Function for image-to-image generation.
    Args:
        init_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, that will be used as the starting point for the
            process.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
            `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
            number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
            noise will be maximum and the denoising process will run for the full number of iterations specified in
            `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference. This parameter will be modulated by `strength`.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    return self.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        max_embeddings_multiples=max_embeddings_multiples,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        **kwargs,
    )

  def inpaint(
      self,
      init_image: Union[torch.FloatTensor, PIL.Image.Image],
      mask_image: Union[torch.FloatTensor, PIL.Image.Image],
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      strength: float = 0.8,
      num_inference_steps: Optional[int] = 50,
      guidance_scale: Optional[float] = 7.5,
      num_images_per_prompt: Optional[int] = 1,
      eta: Optional[float] = 0.0,
      generator: Optional[torch.Generator] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      callback_steps: Optional[int] = 1,
      **kwargs,
  ):
    r"""
    Function for inpaint.
    Args:
        init_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, that will be used as the starting point for the
            process. This is the image whose masked region will be inpainted.
        mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
            replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
            PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
            contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
            is 1, the denoising process will be run on the masked area for the full number of iterations specified
            in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more
            noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
            the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    return self.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        max_embeddings_multiples=max_embeddings_multiples,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        **kwargs,
    )

  # CLIP guidance StableDiffusion
  # copy from https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py

  @torch.enable_grad()
  def cond_fn(self, latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings_clip, clip_guidance_scale,
              num_cutouts, use_cutouts=True, ):
    latents = latents.detach().requires_grad_()

    if isinstance(self.scheduler, LMSDiscreteScheduler):
      sigma = self.scheduler.sigmas[index]
      # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
      latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
    else:
      latent_model_input = latents

    # predict the noise residual
    noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
      alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
      beta_prod_t = 1 - alpha_prod_t
      # compute predicted original sample from predicted noise also called
      # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

      fac = torch.sqrt(beta_prod_t)
      sample = pred_original_sample * (fac) + latents * (1 - fac)
    elif isinstance(self.scheduler, LMSDiscreteScheduler):
      sigma = self.scheduler.sigmas[index]
      sample = latents - sigma * noise_pred
    else:
      raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

    sample = 1 / 0.18215 * sample
    image = self.vae.decode(sample).sample
    image = (image / 2 + 0.5).clamp(0, 1)

    if use_cutouts:
      image = self.make_cutouts(image, num_cutouts)
    else:
      image = transforms.Resize(FEATURE_EXTRACTOR_SIZE)(image)
    image = self.normalize(image).to(latents.dtype)

    image_embeddings_clip = self.clip_model.get_image_features(image)
    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

    if use_cutouts:
      dists = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip)
      dists = dists.view([num_cutouts, sample.shape[0], -1])
      loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
    else:
      loss = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip).mean() * clip_guidance_scale

    grads = -torch.autograd.grad(loss, latents)[0]

    if isinstance(self.scheduler, LMSDiscreteScheduler):
      latents = latents.detach() + grads * (sigma**2)
      noise_pred = noise_pred_original
    else:
      noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
    return noise_pred, latents


class MakeCutouts(torch.nn.Module):
  def __init__(self, cut_size, cut_power=1.0):
    super().__init__()

    self.cut_size = cut_size
    self.cut_power = cut_power

  def forward(self, pixel_values, num_cutouts):
    sideY, sideX = pixel_values.shape[2:4]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, self.cut_size)
    cutouts = []
    for _ in range(num_cutouts):
      size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
      offsetx = torch.randint(0, sideX - size + 1, ())
      offsety = torch.randint(0, sideY - size + 1, ())
      cutout = pixel_values[:, :, offsety: offsety + size, offsetx: offsetx + size]
      cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.cut_size))
    return torch.cat(cutouts)


def spherical_dist_loss(x, y):
  x = torch.nn.functional.normalize(x, dim=-1)
  y = torch.nn.functional.normalize(y, dim=-1)
  return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
  """
  Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
  Accepted tokens are:
    (abc) - increases attention to abc by a multiplier of 1.1
    (abc:3.12) - increases attention to abc by a multiplier of 3.12
    [abc] - decreases attention to abc by a multiplier of 1.1
    \( - literal character '('
    \[ - literal character '['
    \) - literal character ')'
    \] - literal character ']'
    \\ - literal character '\'
    anything else - just text
  >>> parse_prompt_attention('normal text')
  [['normal text', 1.0]]
  >>> parse_prompt_attention('an (important) word')
  [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
  >>> parse_prompt_attention('(unbalanced')
  [['unbalanced', 1.1]]
  >>> parse_prompt_attention('\(literal\]')
  [['(literal]', 1.0]]
  >>> parse_prompt_attention('(unnecessary)(parens)')
  [['unnecessaryparens', 1.1]]
  >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
  [['a ', 1.0],
   ['house', 1.5730000000000004],
   [' ', 1.1],
   ['on', 1.0],
   [' a ', 1.1],
   ['hill', 0.55],
   [', sun, ', 1.1],
   ['sky', 1.4641000000000006],
   ['.', 1.1]]
  """

  res = []
  round_brackets = []
  square_brackets = []

  round_bracket_multiplier = 1.1
  square_bracket_multiplier = 1 / 1.1

  def multiply_range(start_position, multiplier):
    for p in range(start_position, len(res)):
      res[p][1] *= multiplier

  for m in re_attention.finditer(text):
    text = m.group(0)
    weight = m.group(1)

    if text.startswith("\\"):
      res.append([text[1:], 1.0])
    elif text == "(":
      round_brackets.append(len(res))
    elif text == "[":
      square_brackets.append(len(res))
    elif weight is not None and len(round_brackets) > 0:
      multiply_range(round_brackets.pop(), float(weight))
    elif text == ")" and len(round_brackets) > 0:
      multiply_range(round_brackets.pop(), round_bracket_multiplier)
    elif text == "]" and len(square_brackets) > 0:
      multiply_range(square_brackets.pop(), square_bracket_multiplier)
    else:
      res.append([text, 1.0])

  for pos in round_brackets:
    multiply_range(pos, round_bracket_multiplier)

  for pos in square_brackets:
    multiply_range(pos, square_bracket_multiplier)

  if len(res) == 0:
    res = [["", 1.0]]

  # merge runs of identical weights
  i = 0
  while i + 1 < len(res):
    if res[i][1] == res[i + 1][1]:
      res[i][0] += res[i + 1][0]
      res.pop(i + 1)
    else:
      i += 1

  return res


def get_prompts_with_weights(pipe: PipelineLike, prompt: List[str], max_length: int):
  r"""
  Tokenize a list of prompts and return its tokens with weights of each token.
  No padding, starting or ending token is included.
  """
  tokens = []
  weights = []
  truncated = False
  for text in prompt:
    texts_and_weights = parse_prompt_attention(text)
    text_token = []
    text_weight = []
    for word, weight in texts_and_weights:
      # tokenize and discard the starting and the ending token
      token = pipe.tokenizer(word).input_ids[1:-1]
      text_token += token
      # copy the weight by length of token
      text_weight += [weight] * len(token)
      # stop if the text is too long (longer than truncation limit)
      if len(text_token) > max_length:
        truncated = True
        break
    # truncate
    if len(text_token) > max_length:
      truncated = True
      text_token = text_token[:max_length]
      text_weight = text_weight[:max_length]
    tokens.append(text_token)
    weights.append(text_weight)
  if truncated:
    print("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
  return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
  r"""
  Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
  """
  max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
  weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
  for i in range(len(tokens)):
    tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
    if no_boseos_middle:
      weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
    else:
      w = []
      if len(weights[i]) == 0:
        w = [1.0] * weights_length
      else:
        for j in range(max_embeddings_multiples):
          w.append(1.0)  # weight for starting token in this chunk
          w += weights[i][j * (chunk_length - 2): min(len(weights[i]), (j + 1) * (chunk_length - 2))]
          w.append(1.0)  # weight for ending token in this chunk
        w += [1.0] * (weights_length - len(w))
      weights[i] = w[:]

  return tokens, weights


def get_unweighted_text_embeddings(
    pipe: PipelineLike,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
  """
  When the length of tokens is a multiple of the capacity of the text encoder,
  it should be split into chunks and sent to the text encoder individually.
  """
  max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
  if max_embeddings_multiples > 1:
    text_embeddings = []
    for i in range(max_embeddings_multiples):
      # extract the i-th chunk
      text_input_chunk = text_input[:, i * (chunk_length - 2): (i + 1) * (chunk_length - 2) + 2].clone()

      # cover the head and the tail by the starting and the ending tokens
      text_input_chunk[:, 0] = text_input[0, 0]
      if pad == eos:                        # v1
        text_input_chunk[:, -1] = text_input[0, -1]
      else:                                 # v2
        if text_input_chunk[:, -1] != eos and text_input_chunk[:, -1] != pad:     # 最後に普通の文字がある
          text_input_chunk[:, -1] = eos
        if text_input_chunk[:, 1] == pad:                                         # BOSだけであとはPAD
          text_input_chunk[:, 1] = eos

      if clip_skip is None or clip_skip == 1:
        text_embedding = pipe.text_encoder(text_input_chunk)[0]
      else:
        enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
        text_embedding = enc_out['hidden_states'][-clip_skip]
        text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)

      if no_boseos_middle:
        if i == 0:
          # discard the ending token
          text_embedding = text_embedding[:, :-1]
        elif i == max_embeddings_multiples - 1:
          # discard the starting token
          text_embedding = text_embedding[:, 1:]
        else:
          # discard both starting and ending tokens
          text_embedding = text_embedding[:, 1:-1]

      text_embeddings.append(text_embedding)
    text_embeddings = torch.concat(text_embeddings, axis=1)
  else:
    if clip_skip is None or clip_skip == 1:
      text_embeddings = pipe.text_encoder(text_input)[0]
    else:
      enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
      text_embeddings = enc_out['hidden_states'][-clip_skip]
      text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
  return text_embeddings


def get_weighted_text_embeddings(
    pipe: PipelineLike,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
    **kwargs,
):
  r"""
  Prompts can be assigned with local weights using brackets. For example,
  prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
  and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
  Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
  Args:
      pipe (`DiffusionPipeline`):
          Pipe to provide access to the tokenizer and the text encoder.
      prompt (`str` or `List[str]`):
          The prompt or prompts to guide the image generation.
      uncond_prompt (`str` or `List[str]`):
          The unconditional prompt or prompts for guide the image generation. If unconditional prompt
          is provided, the embeddings of prompt and uncond_prompt are concatenated.
      max_embeddings_multiples (`int`, *optional*, defaults to `1`):
          The max multiple length of prompt embeddings compared to the max output length of text encoder.
      no_boseos_middle (`bool`, *optional*, defaults to `False`):
          If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
          ending token in each of the chunk in the middle.
      skip_parsing (`bool`, *optional*, defaults to `False`):
          Skip the parsing of brackets.
      skip_weighting (`bool`, *optional*, defaults to `False`):
          Skip the weighting. When the parsing is skipped, it is forced True.
  """
  max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
  if isinstance(prompt, str):
    prompt = [prompt]

  if not skip_parsing:
    prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
    if uncond_prompt is not None:
      if isinstance(uncond_prompt, str):
        uncond_prompt = [uncond_prompt]
      uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
  else:
    prompt_tokens = [
        token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
    ]
    prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
    if uncond_prompt is not None:
      if isinstance(uncond_prompt, str):
        uncond_prompt = [uncond_prompt]
      uncond_tokens = [
          token[1:-1]
          for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
      ]
      uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

  # round up the longest length of tokens to a multiple of (model_max_length - 2)
  max_length = max([len(token) for token in prompt_tokens])
  if uncond_prompt is not None:
    max_length = max(max_length, max([len(token) for token in uncond_tokens]))

  max_embeddings_multiples = min(
      max_embeddings_multiples,
      (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
  )
  max_embeddings_multiples = max(1, max_embeddings_multiples)
  max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

  # pad the length of tokens and weights
  bos = pipe.tokenizer.bos_token_id
  eos = pipe.tokenizer.eos_token_id
  pad = pipe.tokenizer.pad_token_id
  prompt_tokens, prompt_weights = pad_tokens_and_weights(
      prompt_tokens,
      prompt_weights,
      max_length,
      bos,
      eos,
      pad,
      no_boseos_middle=no_boseos_middle,
      chunk_length=pipe.tokenizer.model_max_length,
  )
  prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
  if uncond_prompt is not None:
    uncond_tokens, uncond_weights = pad_tokens_and_weights(
        uncond_tokens,
        uncond_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

  # get the embeddings
  text_embeddings = get_unweighted_text_embeddings(
      pipe,
      prompt_tokens,
      pipe.tokenizer.model_max_length,
      clip_skip,
      eos, pad,
      no_boseos_middle=no_boseos_middle,
  )
  prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
  if uncond_prompt is not None:
    uncond_embeddings = get_unweighted_text_embeddings(
        pipe,
        uncond_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        eos, pad,
        no_boseos_middle=no_boseos_middle,
    )
    uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

  # assign weights to the prompts and normalize in the sense of mean
  # TODO: should we normalize by chunk or in a whole (current implementation)?
  # →全体でいいんじゃないかな
  if (not skip_parsing) and (not skip_weighting):
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings *= prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
    if uncond_prompt is not None:
      previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
      uncond_embeddings *= uncond_weights.unsqueeze(-1)
      current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
      uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

  if uncond_prompt is not None:
    return text_embeddings, uncond_embeddings, prompt_tokens
  return text_embeddings, None, prompt_tokens


def preprocess_guide_image(image):
  image = image.resize(FEATURE_EXTRACTOR_SIZE, resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32) / 255.0
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  return image                              # 0 to 1


def preprocess_image(image):
  w, h = image.size
  w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32) / 255.0
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  return 2.0 * image - 1.0


def preprocess_mask(mask):
  mask = mask.convert("L")
  w, h = mask.size
  w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
  mask = mask.resize((w // 8, h // 8), resample=PIL.Image.LANCZOS)
  mask = np.array(mask).astype(np.float32) / 255.0
  mask = np.tile(mask, (4, 1, 1))
  mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
  mask = 1 - mask  # repaint white, keep black
  mask = torch.from_numpy(mask)
  return mask


# endregion

VAE_PREFIX = "first_stage_model."


def load_vae(vae, dtype):
  print(f"load VAE: {vae}")
  if os.path.isdir(vae) or not os.path.isfile(vae):
    # Diffusers
    if not os.path.isdir(vae):      # load from Hugging Face
      subfolder = "vae"
    else:
      subfolder = None
    vae = AutoencoderKL.from_pretrained(vae, subfolder=subfolder, torch_dtype=dtype)
    return vae

  vae_config = create_vae_diffusers_config()

  if vae.endswith(".bin"):
    # SD 1.5 VAE on Huggingface
    vae_sd = torch.load(vae, map_location="cpu")
    converted_vae_checkpoint = vae_sd
  else:
    # StableDiffusion
    vae_model = torch.load(vae, map_location="cpu")
    vae_sd = vae_model['state_dict']

    # vae only or full model
    full_model = False
    for vae_key in vae_sd:
      if vae_key.startswith(VAE_PREFIX):
        full_model = True
        break
    if not full_model:
      sd = {}
      for key, value in vae_sd.items():
        sd[VAE_PREFIX + key] = value
      vae_sd = sd
      del sd

    # Convert the VAE model.
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_sd, vae_config)

  vae = AutoencoderKL(**vae_config)
  vae.load_state_dict(converted_vae_checkpoint)
  return vae


def main(args):
  if args.fp16:
    dtype = torch.float16
  elif args.bf16:
    dtype = torch.bfloat16
  else:
    dtype = torch.float32

  highres_fix = args.highres_fix_scale is not None
  assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

  assert not args.v2 or (args.sampler in ['ddim', 'euler', 'k_euler']
                         ), f"only ddim/euler supported for SDv2 / SDv2ではsamplerはddimかeulerしか使えません"

  if args.v_parameterization and not args.v2:
    print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
  if args.v2 and args.clip_skip is not None:
    print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

  # モデルを読み込む
  use_stable_diffusion_format = os.path.isfile(args.ckpt)
  if use_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(args.v2, args.ckpt)
  else:
    print("load Diffusers pretrained models")
    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=dtype)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    del pipe

  # VAEを読み込む
  if args.vae is not None:
    vae = load_vae(args.vae, dtype)
    print("additional VAE loaded")

  if args.clip_guidance_scale > 0.0 or args.clip_image_guidance_scale:
    print("prepare clip model")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=dtype)
  else:
    clip_model = None

  # xformers、Hypernetwork対応
  if not args.diffusers_xformers:
    replace_unet_modules(unet, not args.xformers, args.xformers)

  # hypernetworkを組み込む
  if args.hypernetwork_module is not None:
    assert not args.diffusers_xformers, "cannot use hypernetwork with diffusers_xformers / diffusers_xformers指定時はHypernetworkは利用できません"

    print("import hypernetwork module:", args.hypernetwork_module)
    hyp_module = importlib.import_module(args.hypernetwork_module)

    hypernetwork = hyp_module.Hypernetwork(args.hypernetwork_mul)

    print("load hypernetwork weights from:", args.hypernetwork_weights)
    hyp_sd = torch.load(args.hypernetwork_weights, map_location='cpu')
    success = hypernetwork.load_from_state_dict(hyp_sd)
    assert success, "hypernetwork weights loading failed."

    if args.opt_channels_last:
      hypernetwork.to(memory_format=torch.channels_last)
  else:
    hypernetwork = None

  # tokenizerを読み込む
  print("loading tokenizer")
  if use_stable_diffusion_format:
    if args.v2:
      tokenizer = CLIPTokenizer.from_pretrained(V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
    else:
      tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)  # , model_max_length=max_token_length + 2)

  # schedulerを用意する
  sched_init_args = {}
  if args.sampler == "ddim":
    scheduler_cls = DDIMScheduler
    scheduler_module = diffusers.schedulers.scheduling_ddim
  elif args.sampler == "ddpm":                    # ddpmはおかしくなるのでoptionから外してある
    scheduler_cls = DDPMScheduler
    scheduler_module = diffusers.schedulers.scheduling_ddpm
  elif args.sampler == "pndm":
    scheduler_cls = PNDMScheduler
    scheduler_module = diffusers.schedulers.scheduling_pndm
  elif args.sampler == 'lms' or args.sampler == 'k_lms':
    scheduler_cls = LMSDiscreteScheduler
    scheduler_module = diffusers.schedulers.scheduling_lms_discrete
  elif args.sampler == 'euler' or args.sampler == 'k_euler':
    scheduler_cls = EulerDiscreteScheduler
    scheduler_module = diffusers.schedulers.scheduling_euler_discrete
  elif args.sampler == 'euler_a' or args.sampler == 'k_euler_a':
    scheduler_cls = EulerAncestralDiscreteScheduler
    scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
  elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
    scheduler_cls = DPMSolverMultistepScheduler
    sched_init_args['algorithm_type'] = args.sampler
    scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep

  if args.v_parameterization:
    sched_init_args['prediction_type'] = 'v_prediction'

  # samplerの乱数をあらかじめ指定するための処理

  # replace randn
  class NoiseManager:
    def __init__(self):
      self.sampler_noises = None
      self.sampler_noise_index = 0

    def reset_sampler_noises(self, noises):
      self.sampler_noise_index = 0
      self.sampler_noises = noises

    def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
      # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
      if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
        noise = self.sampler_noises[self.sampler_noise_index]
        if shape != noise.shape:
          noise = None
      else:
        noise = None

      if noise == None:
        print(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
        noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

      self.sampler_noise_index += 1
      return noise

  class TorchRandReplacer:
    def __init__(self, noise_manager):
      self.noise_manager = noise_manager

    def __getattr__(self, item):
      if item == 'randn':
        return self.noise_manager.randn
      if hasattr(torch, item):
        return getattr(torch, item)
      raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

  noise_manager = NoiseManager()
  if scheduler_module is not None:
    scheduler_module.torch = TorchRandReplacer(noise_manager)

  scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS,
                            beta_start=SCHEDULER_LINEAR_START, beta_end=SCHEDULER_LINEAR_END,
                            beta_schedule=SCHEDLER_SCHEDULE, **sched_init_args)

  # custom pipelineをコピったやつを生成する
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")             # "mps"を考量してない
  vae.to(dtype).to(device)
  text_encoder.to(dtype).to(device)
  unet.to(dtype).to(device)
  if clip_model is not None:
    clip_model.to(dtype).to(device)

  if hypernetwork is not None:
    hypernetwork.to(dtype).to(device)
    print("apply hypernetwork")
    hypernetwork.apply_to_diffusers(vae, text_encoder, unet)

  if args.opt_channels_last:
    print(f"set optimizing: channels last")
    text_encoder.to(memory_format=torch.channels_last)
    vae.to(memory_format=torch.channels_last)
    unet.to(memory_format=torch.channels_last)
    if clip_model is not None:
      clip_model.to(memory_format=torch.channels_last)
    if hypernetwork is not None:
      hypernetwork.to(memory_format=torch.channels_last)

  pipe = PipelineLike(device, vae, text_encoder, tokenizer, unet, scheduler, args.clip_skip,
                      clip_model, args.clip_guidance_scale, args.clip_image_guidance_scale)
  print("pipeline is ready.")

  if args.diffusers_xformers:
    pipe.enable_xformers_memory_efficient_attention()

  # promptを取得する
  if args.from_file is not None:
    print(f"reading prompts from {args.from_file}")
    with open(args.from_file, "r", encoding="utf-8") as f:
      prompt_list = f.read().splitlines()
      prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
  elif args.prompt is not None:
    prompt_list = [args.prompt]
  else:
    prompt_list = []

  if args.interactive:
    args.n_iter = 1

  # img2imgの前処理、画像の読み込みなど
  def load_images(path):
    if os.path.isfile(path):
      paths = [path]
    else:
      paths = glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")) + \
          glob.glob(os.path.join(path, "*.jpeg"))

    images = []
    for p in paths:
      image = Image.open(p)
      if image.mode != "RGB":
        print(f"convert image to RGB from {image.mode}: {p}")
        image = image.convert("RGB")
      images.append(image)
    return images

  def resize_images(imgs, size):
    resized = []
    for img in imgs:
      resized.append(img.resize(size, Image.Resampling.LANCZOS))
    return resized

  if args.image_path is not None:
    print(f"load image for img2img: {args.image_path}")
    init_images = load_images(args.image_path)
    assert len(init_images) > 0, f"No image / 画像がありません: {args.image_path}"
    print(f"loaded {len(init_images)} images for img2img")
  else:
    init_images = None

  if args.mask_path is not None:
    print(f"load mask for inpainting: {args.mask_path}")
    mask_images = load_images(args.mask_path)
    assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {args.image_path}"
    print(f"loaded {len(mask_images)} mask images for inpainting")
  else:
    mask_images = None

  # promptがないとき、画像のPngInfoから取得する
  if init_images is not None and len(prompt_list) == 0 and not args.interactive:
    print("get prompts from images' meta data")
    for img in init_images:
      if 'prompt' in img.text:
        prompt = img.text['prompt']
        if 'negative-prompt' in img.text:
          prompt += " --n " + img.text['negative-prompt']
        prompt_list.append(prompt)

    # 指定回数だけ繰り返す
    l = []
    for p in prompt_list:
      l.extend([p] * args.images_per_prompt)
    prompt_list = l

    l = []
    for im in init_images:
      l.extend([im] * args.images_per_prompt)
    init_images = l

    if mask_images is not None:
      l = []
      for im in mask_images:
        l.extend([im] * args.images_per_prompt)
      mask_images = l

  if init_images is not None and args.W is not None and args.H is not None:
    print(f"resize img2img source images to {args.W}*{args.H}")
    init_images = resize_images(init_images, (args.W, args.H))
    if mask_images is not None:
      print(f"resize img2img mask images to {args.W}*{args.H}")
      mask_images = resize_images(mask_images, (args.W, args.H))

  if args.guide_image_path is not None:
    print(f"load image for CLIP guidance: {args.guide_image_path}")
    guide_images = load_images(args.guide_image_path)
    assert len(guide_images) > 0, f"No guide image / ガイド画像がありません: {args.image_path}"
    print(f"loaded {len(guide_images)} guide images for CLIP guidance")
  else:
    guide_images = None

  # seed指定時はseedを決めておく
  if args.seed is not None:
    random.seed(args.seed)
    predefined_seeds = [random.randint(0, 0x7fffffff) for _ in range(args.n_iter * len(prompt_list) * args.images_per_prompt)]
    if len(predefined_seeds) == 1:
      predefined_seeds[0] = args.seed
  else:
    predefined_seeds = None

  # デフォルト画像サイズを設定する：img2imgではこれらの値は無視される（またはW*Hにリサイズ済み）
  if args.W is None:
    args.W = 512
  if args.H is None:
    args.H = 512

  # 画像生成のループ
  os.makedirs(args.outdir, exist_ok=True)
  max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

  for iter in range(args.n_iter):
    print(f"iteration {iter+1}/{args.n_iter}")

    # バッチ処理の関数
    def process_batch(batch, highres_fix, highres_1st=False):
      batch_size = len(batch)

      # highres_fixの処理
      if highres_fix and not highres_1st:
        # 1st stageのバッチを作成して呼び出す
        print("process 1st stage1")
        batch_1st = []
        for params1, (width, height, steps, scale, strength) in batch:
          width_1st = int(width * args.highres_fix_scale + .5)
          height_1st = int(height * args.highres_fix_scale + .5)
          width_1st = width_1st - width_1st % 32
          height_1st = height_1st - height_1st % 32
          batch_1st.append((params1, (width_1st, height_1st, args.highres_fix_steps, scale, strength)))
        images_1st = process_batch(batch_1st, True, True)

        # 2nd stageのバッチを作成して以下処理する
        print("process 2nd stage1")
        batch_2nd = []
        for i, (b1, image) in enumerate(zip(batch, images_1st)):
          image = image.resize((width, height), resample=PIL.Image.LANCZOS)
          (step, prompt, negative_prompt, seed, _, _, clip_prompt, guide_image), params2 = b1
          batch_2nd.append(((step, prompt, negative_prompt, seed+1, image, None, clip_prompt, guide_image), params2))
        batch = batch_2nd

      (step_first, _, _, _, init_image, mask_image, _, guide_image), (width, height, steps, scale, strength) = batch[0]
      noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

      prompts = []
      negative_prompts = []
      start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
      noises = [torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype) for _ in range(steps)]
      seeds = []
      clip_prompts = []

      if init_image is not None:                      # img2img?
        i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
        init_images = []

        if mask_image is not None:
          mask_images = []
        else:
          mask_images = None
      else:
        i2i_noises = None
        init_images = None
        mask_images = None

      if guide_image is not None:                     # CLIP image guided?
        guide_images = []
      else:
        guide_images = None

      # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
      all_images_are_same = True
      all_masks_are_same = True
      all_guide_images_are_same = True
      for i, ((_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image), _) in enumerate(batch):
        prompts.append(prompt)
        negative_prompts.append(negative_prompt)
        seeds.append(seed)
        clip_prompts.append(clip_prompt)

        if init_image is not None:
          init_images.append(init_image)
          if i > 0 and all_images_are_same:
            all_images_are_same = init_images[-2] is init_image

        if mask_image is not None:
          mask_images.append(mask_image)
          if i > 0 and all_masks_are_same:
            all_masks_are_same = mask_images[-2] is mask_image

        if guide_image is not None:
          guide_images.append(guide_image)
          if i > 0 and all_guide_images_are_same:
            all_guide_images_are_same = guide_images[-2] is guide_image

        # make start code
        torch.manual_seed(seed)
        start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

        # make each noises
        for j in range(steps):
          noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

        if i2i_noises is not None:                # img2img noise
          i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

      noise_manager.reset_sampler_noises(noises)

      # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
      if init_images is not None and all_images_are_same:
        init_images = init_images[0]
      if mask_images is not None and all_masks_are_same:
        mask_images = mask_images[0]
      if guide_images is not None and all_guide_images_are_same:
        guide_images = guide_images[0]

      # generate
      images = pipe(prompts, negative_prompts, init_images, mask_images, height, width, steps, scale, strength, latents=start_code,
                    output_type='pil', max_embeddings_multiples=max_embeddings_multiples, img2img_noise=i2i_noises, clip_prompts=clip_prompts, clip_guide_images=guide_images)[0]
      if highres_1st and not args.highres_fix_save_1st:
        return images

      # save image
      highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
      ts_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
      for i, (image, prompt, negative_prompts, seed, clip_prompt) in enumerate(zip(images, prompts, negative_prompts, seeds, clip_prompts)):
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("sampler", args.sampler)
        metadata.add_text("steps", str(steps))
        metadata.add_text("scale", str(scale))
        if negative_prompt is not None:
          metadata.add_text("negative-prompt", negative_prompt)
        if clip_prompt is not None:
          metadata.add_text("clip-prompt", clip_prompt)

        fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png" if args.sequential_file_name else f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"
        image.save(os.path.join(args.outdir, fln), pnginfo=metadata)

      if not args.no_preview and not highres_1st and args.interactive:
        try:
          import cv2
          for prompt, image in zip(prompts, images):
            cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])      # プロンプトが長いと死ぬ
            cv2.waitKey()
            cv2.destroyAllWindows()
        except ImportError:
          print("opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません")

      return images

    # 画像生成のプロンプトが一周するまでのループ
    prompt_index = 0
    global_step = 0
    batch_data = []
    while args.interactive or prompt_index < len(prompt_list):
      if len(prompt_list) == 0:
        # interactive
        valid = False
        while not valid:
          print("\nType prompt:")
          try:
            prompt = input()
          except EOFError:
            break

          valid = len(prompt.strip().split(' --')[0].strip()) > 0
        if not valid:                                     # EOF, end app
          break
      else:
        prompt = prompt_list[prompt_index]

      # parse prompt
      width = args.W
      height = args.H
      scale = args.scale
      steps = args.steps
      seeds = None
      strength = 0.8 if args.strength is None else args.strength
      negative_prompt = ""
      clip_prompt = None

      prompt_args = prompt.strip().split(' --')
      prompt = prompt_args[0]
      print(f"prompt {prompt_index+1}/{len(prompt_list)}: {prompt}")

      for parg in prompt_args[1:]:
        try:
          m = re.match(r'w (\d+)', parg)
          if m:
            width = int(m.group(1))
            print(f"width: {width}")
            continue

          m = re.match(r'h (\d+)', parg)
          if m:
            height = int(m.group(1))
            print(f"height: {height}")
            continue

          m = re.match(r's (\d+)', parg)
          if m:               # steps
            steps = max(1, min(1000, int(m.group(1))))
            print(f"steps: {steps}")
            continue

          m = re.match(r'd ([\d,]+)', parg)
          if m:               # seed
            seeds = [int(d) for d in m.group(1).split(',')]
            print(f"seeds: {seeds}")
            continue

          m = re.match(r'l ([\d\.]+)', parg)
          if m:               # scale
            scale = float(m.group(1))
            print(f"scale: {scale}")
            continue

          m = re.match(r't ([\d\.]+)', parg)
          if m:               # strength
            strength = float(m.group(1))
            print(f"strength: {strength}")
            continue

          m = re.match(r'n (.+)', parg)
          if m:               # negative prompt
            negative_prompt = m.group(1)
            print(f"negative prompt: {negative_prompt}")
            continue

          m = re.match(r'c (.+)', parg)
          if m:               # negative prompt
            clip_prompt = m.group(1)
            print(f"clip prompt: {clip_prompt}")
            continue
        except ValueError as ex:
          print(f"Exception in parsing / 解析エラー: {parg}")
          print(ex)

      if seeds is not None:
        # 数が足りないなら繰り返す
        if len(seeds) < args.images_per_prompt:
          seeds = seeds * int(math.ceil(args.images_per_prompt / len(seeds)))
        seeds = seeds[:args.images_per_prompt]
      else:
        if predefined_seeds is not None:
          seeds = predefined_seeds[-args.images_per_prompt:]
          predefined_seeds = predefined_seeds[:-args.images_per_prompt]
        else:
          seeds = [random.randint(0, 0x7fffffff) for _ in range(args.images_per_prompt)]
        if args.interactive:
          print(f"seed: {seeds}")

      init_image = mask_image = guide_image = None
      for seed in seeds:                  # images_per_promptの数だけ
        # 同一イメージを使うとき、本当はlatentに変換しておくと無駄がないが面倒なのでとりあえず毎回処理する
        if init_images is not None:
          init_image = init_images[global_step % len(init_images)]

          # 32単位に丸めたやつにresizeされるので踏襲する
          width, height = init_image.size
          width = width - width % 32
          height = height - height % 32
          if width != init_image.size[0] or height != init_image.size[1]:
            print(f"img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます")

        if mask_images is not None:
          mask_image = mask_images[global_step % len(mask_images)]

        if guide_images is not None:
          guide_image = guide_images[global_step % len(guide_images)]

        b1 = ((global_step, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image),
              (width, height, steps, scale, strength))
        if len(batch_data) > 0 and batch_data[-1][1] != b1[1]:  # バッチ分割必要？
          process_batch(batch_data)
          batch_data.clear()

        batch_data.append(b1)
        if len(batch_data) == args.batch_size:
          process_batch(batch_data, highres_fix)
          batch_data.clear()

        global_step += 1

      prompt_index += 1

    if len(batch_data) > 0:
      process_batch(batch_data, highres_fix)
      batch_data.clear()

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--v2", action='store_true', help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
  parser.add_argument("--v_parameterization", action='store_true',
                      help='enable v-parameterization training / v-parameterization学習を有効にする')
  parser.add_argument("--prompt", type=str, default=None, help="prompt / プロンプト")
  parser.add_argument("--from_file", type=str, default=None,
                      help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む")
  parser.add_argument("--interactive", action='store_true', help='interactive mode (generates one image) / 対話モード（生成される画像は1枚になります）')
  parser.add_argument("--no_preview", action='store_true', help='do not show generated image in interactive mode / 対話モードで画像を表示しない')
  parser.add_argument("--image_path", type=str, default=None, help="image to inpaint or to generate from / img2imgまたはinpaintを行う元画像")
  parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting / inpaint時のマスク")
  parser.add_argument("--strength", type=float, default=None, help="img2img strength / img2img時のstrength")
  parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt / プロンプトあたりの出力枚数")
  parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
  parser.add_argument("--sequential_file_name", action='store_true',  help="sequential output file name / 生成画像のファイル名を連番にする")
  # parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
  parser.add_argument("--n_iter", type=int, default=1, help="sample this often / 繰り返し回数")
  parser.add_argument("--H", type=int, default=None, help="image height, in pixel space / 生成画像高さ")
  parser.add_argument("--W", type=int, default=None, help="image width, in pixel space / 生成画像幅")
  parser.add_argument("--batch_size", type=int, default=1, help="batch size / バッチサイズ")
  parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps / サンプリングステップ数")
  parser.add_argument('--sampler', type=str, default='ddim',
                      choices=['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'dpmsolver', 'dpmsolver++', 'k_lms', 'k_euler', 'k_euler_a'], help=f'sampler (scheduler) type / サンプラー（スケジューラ）の種類')
  parser.add_argument("--scale", type=float, default=7.5,
                      help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale")
  parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint of model / モデルのcheckpointファイルまたはディレクトリ")
  parser.add_argument("--vae", type=str, default=None,
                      help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
  parser.add_argument("--seed", type=int, default=None,
                      help="seed, or seed of seeds in multiple generation / 1枚生成時のseed、または複数枚生成時の乱数seedを決めるためのseed")
  parser.add_argument("--fp16", action='store_true', help='use fp16 / fp16を指定し省メモリ化する')
  parser.add_argument("--bf16", action='store_true', help='use bfloat16 / bfloat16を指定し省メモリ化する')
  parser.add_argument("--xformers", action='store_true', help='use xformers / xformersを使用し高速化する')
  parser.add_argument("--diffusers_xformers", action='store_true',
                      help='use xformers by diffusers (Hypernetworks doen\'t work) / Diffusersでxformersを使用する（Hypernetwork利用不可）')
  parser.add_argument("--opt_channels_last", action='store_true',
                      help='set channels last option to model / モデルにchannles lastを指定し最適化する')
  parser.add_argument("--hypernetwork_module", type=str, default=None, help='Hypernetwork module to use / Hypernetworkを使う時そのモジュール名')
  parser.add_argument("--hypernetwork_weights", type=str, default=None, help='Hypernetwork weights to load / Hypernetworkの重み')
  parser.add_argument("--hypernetwork_mul", type=float, default=1.0, help='Hypernetwork multiplier / Hypernetworkの効果の倍率')
  parser.add_argument("--clip_skip", type=int, default=None, help='layer number from bottom to use in CLIP / CLIPの後ろからn層目の出力を使う')
  parser.add_argument("--max_embeddings_multiples", type=int, default=None,
                      help='max embeding multiples, max token length is 75 * multiples / トークン長をデフォルトの何倍とするか 75*この値 がトークン長となる')
  parser.add_argument("--clip_guidance_scale", type=float, default=0.0,
                      help='enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only) / CLIP guided SDを有効にしてこのscaleを適用する（サンプラーはDDIM、PNDM、LMSのみ）')
  parser.add_argument("--clip_image_guidance_scale", type=float, default=0.0,
                      help='enable CLIP guided SD by image, scale for guidance / 画像によるCLIP guided SDを有効にしてこのscaleを適用する')
  parser.add_argument("--guide_image_path", type=str, default=None, help="image to CLIP guidance / CLIP guided SDでガイドに使う画像")
  parser.add_argument("--highres_fix_scale", type=float, default=None,
                      help="enable highres fix, reso scale for 1st stage / highres fixを有効にして最初の解像度をこのscaleにする")
  parser.add_argument("--highres_fix_steps", type=int, default=28,
                      help="1st stage steps for highres fix / highres fixの最初のステージのステップ数")
  parser.add_argument("--highres_fix_save_1st", action='store_true',
                      help="save 1st stage images for highres fix / highres fixの最初のステージの画像を保存する")

  args = parser.parse_args()
  main(args)
