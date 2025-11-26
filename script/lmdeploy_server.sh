#!/bin/bash

pip install lmdeploy timm peft>=0.17.0 openai
CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server OpenGVLab/InternVL3-8B --chat-template internvl2_5 --server-port 23333 --tp 1 # takes 30GB vram.
