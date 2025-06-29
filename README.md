# EXPOTION: FACIAL EXPRESSION AND MOTION CONTROL FOR MULTIMODAL MUSIC GENERATION

A multimodal music generation project using video and audio inputs.

## Demo Page

🔗 [View the live demo](https://github.com/fathinah/expotion)

## Dataset

We use the [Human Music Moves dataset (video & audio)](https://huggingface.co/datasets/mbzmusic/human-music-moves).

## Features

1. **SynchFormer**  
   Precomputed control signals at 5 fps:  
   https://huggingface.co/datasets/mbzmusic/synchformer_5fps/tree/main

2. **RAFT**  
   Optical‐flow features at 5 fps:  
   https://huggingface.co/datasets/mbzmusic/output_raft_nocap_5fps_v1

## Requirements

- Python **3.11**  
- See [requirements.txt](requirements.txt)

## Installation

```bash
# Create & activate your Python 3.11 environment, then:
pip install -r requirements.txt
